import math
import torch
import torch.nn as nn

from pyutils.util_env import UtilEnv


class AttentionPoolClassifier(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=128, num_classes=2):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, L, D]
        h = torch.tanh(self.feature_proj(x))  # [B, L, H]
        scores = self.attn_score(h)  # [B, L, 1]
        attn_weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(attn_weights * h, dim=1)  # [B, H]
        logits = self.classifier(pooled)  # [B, 2]
        return logits


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)
        y = self.fc(y).view(b, 1, c)
        return x * y


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # max_len 最大是5000个时间步
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  # 5000行， d_model 的列的0向量
        pe.require_grad = False  # 位置编码不需要训练

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # div_term=e^(arange(0,d_model,2)* -log(10000.0)/d_model)=

        # Debugger.plot_ts(div_term)

        # 所有的行，偶数列赋值
        pe[:, 0::2] = torch.sin(position * div_term)

        # 所有行，奇数列赋值
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # 奇数：cos 少一列
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # 偶数：正常赋值

        # Debugger.plot_2d_value(torch.sin(position * div_term).numpy())
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiKernelConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: list, dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        out_per_kernel = max(1, out_ch // len(kernel_sizes)) # 每个kernel 输出的数量,

        for k in kernel_sizes:
            padding = dilation * (k - 1) // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_per_kernel, k, padding=padding, bias=False, dilation=dilation),
                nn.BatchNorm1d(out_per_kernel),
                nn.ReLU(inplace=True),
            ))

        self.out_ch = out_per_kernel * len(kernel_sizes)
        self.se = SEModule(self.out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        out = self.se(out)
        return out


class Model(nn.Module):
    """M2SFormer-TS v74: Dilated + positional embedding"""

    def __init__(self, configs):
        super().__init__()
        assert configs.task_name == 'classification'
        # >>>>>>>>>>>环境变量获取的值
        dropout = float(UtilEnv.get_value("HP__DROPOUT", 0.01))
        hidden_dim = int(UtilEnv.get_value("HP__HIDDEN_DIM", 128))
        seq_len = int(UtilEnv.get_value("HP__SEQ_LEN", 5000))
        d_model = int(UtilEnv.get_value("HP__D_MODEL", 128))
        # <<<<<<<<<<环境变量获取的值

        enc_in = configs.enc_in
        num_class = configs.num_class
        # d_model = configs.d_model
        # seq_len = configs.seq_len
        print("✅✅✅Model Conf", dropout, hidden_dim, seq_len, d_model)
        # sys.exit(0)
        self.poisition_embeding = PositionalEmbedding(d_model, max_len=seq_len)
        # Input projection: enc_in -> d_model
        self.input_proj = nn.Linear(enc_in, d_model)

        # Stage 1: dilation=1 (input is now d_model channels)
        self.conv1 = MultiKernelConv(d_model, 96, kernel_sizes=[15, 19, 23], dropout=dropout, dilation=1)
        self.norm1 = nn.LayerNorm(self.conv1.out_ch)

        # Stage 2: dilation=4
        self.conv2 = MultiKernelConv(self.conv1.out_ch, 128, kernel_sizes=[11, 13, 15], dropout=dropout, dilation=4)
        self.norm2 = nn.LayerNorm(self.conv2.out_ch)

        # Stage 3: dilation=16
        self.conv3 = MultiKernelConv(self.conv2.out_ch, 160, kernel_sizes=[7, 9, 11], dropout=dropout, dilation=16)
        self.norm3 = nn.LayerNorm(self.conv3.out_ch)

        # Stage 4: dilation=32
        self.conv4 = MultiKernelConv(self.conv3.out_ch, 128, kernel_sizes=[3, 5, 7], dropout=dropout, dilation=32)
        self.norm4 = nn.LayerNorm(self.conv4.out_ch)
        self.final_dropout = nn.Dropout(dropout)

        self.attention_pool = AttentionPoolClassifier(
            input_dim=self.conv4.out_ch,
            hidden_dim=hidden_dim,
            num_classes=num_class
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # save_pkl(x_enc[0], "x_origin.pkl")
        # save_pkl(self.poisition_embeding(x_enc)[0], "x_origin_position.pkl")
        x = self.input_proj(x_enc) + self.poisition_embeding(x_enc)

        x = self.conv1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.final_dropout(x)
        output = self.attention_pool(x)
        return output
