# KingMomentum ETF / LOF Streamlit App

这是一个可独立运行和部署的 KingMomentum ETF/LOF 轮动策略应用。项目不依赖 QuantSpace 主项目，代码、依赖和9个标的的调整后日线数据都在本目录内。

## 应用功能

- **回测**：选择标的池、回测区间和仓位模式，点击“开始回测”后计算；支持最近1/2/3/5年快捷区间；
- **仓位模式**：均衡仓位（目标波动率20%）、防守仓位（目标波动率15%）、原始满仓/现金，以及自定义目标波动率和仓位调整带；
- **绩效与图表**：累计收益、年化收益、最大回撤、夏普系数、交易次数、净值曲线和反向回撤曲线；
- **记录**：信号调仓、风险仓位再平衡、每日净值和每日持仓，支持CSV下载；
- **周期收益**：月度热力图、季度收益和年度收益；红色代表盈利，绿色代表亏损；
- **最新持仓**：计算最新动量分数、推荐标的、目标总仓位、当前策略仓位和仓位调整判断；
- **策略说明**：解释25日加权回归、动量分数、过热阈值、换仓缓冲、目标波动率和调仓记录公式，并提供实际数据案例。

## 项目结构

```text
KingMomentum_Streamlit_App/
├── app.py                         # Streamlit 页面
├── kingmomentum_core.py           # 评分、仓位管理和回测引擎
├── data/*.parquet                 # 9个标的的调整后日线快照
├── scripts/validate_app.py        # 上传/部署前自检
├── open_kingmomentum_app.command  # macOS双击启动并打开浏览器
├── requirements.txt               # Streamlit运行依赖
├── .gitignore                     # 密钥、虚拟环境和缓存排除规则
└── README.md
```

## 本地运行

推荐先运行部署前自检：

```bash
cd KingMomentum_Streamlit_App
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/validate_app.py
```

随后启动：

```bash
.venv/bin/streamlit run app.py
```

macOS 也可以双击 `open_kingmomentum_app.command`。脚本会检查依赖、启动服务并自动打开 `http://127.0.0.1:8501`；关闭终端即可停止服务。

## 上传 GitHub

将本目录作为 GitHub 仓库根目录，上传以下内容即可：

- `app.py`
- `kingmomentum_core.py`
- `data/`
- `scripts/`
- `requirements.txt`
- `open_kingmomentum_app.command`
- `README.md`
- `.gitignore`

不要上传：

- `.venv/`
- `__pycache__/`
- `.ruff_cache/`
- `.streamlit/secrets.toml`
- `.env` 或其他账号密码文件
- `streamlit.log`

上传前执行：

```bash
.venv/bin/python scripts/validate_app.py
```

## Streamlit 部署

在 Streamlit Cloud 中：

1. 选择 GitHub 仓库和分支；
2. 将主文件设置为 `app.py`；
3. 使用 `requirements.txt` 安装依赖；
4. 部署后检查“回测”“最新持仓”和“策略说明”三个页面。

基础应用使用仓库内置数据快照，不需要 PandaData 账号即可运行回测。如果要启用“最新持仓 → 更新数据”，还需要部署环境能够安装并访问 PandaData SDK，并在 Streamlit 的 `Settings → Secrets` 中配置：

```toml
PANDA_DATA_USERNAME = "你的PandaData账号"
PANDA_DATA_PASSWORD = "你的PandaData密码"
```

本地配置文件路径为：

```text
.streamlit/secrets.toml
```

当前 `requirements.txt` 刻意不包含 `panda_data`，因为该 SDK 不一定能从公开 PyPI 安装。若部署环境无法安装 SDK，网页仍可使用内置数据快照，“更新数据”按钮会给出失败提示，不会伪造更新成功。

## 策略和数据边界

- 动量信号使用过去25个交易日的对数复权收盘价进行加权线性回归；
- 分数小于等于0的标的不参与有效候选；分数超过500视为过热；换仓缓冲为5；
- 信号在收盘后计算，实际交易在下一交易日开盘执行；
- 默认回测最早日期为2017-08-01，尚未上市的标的不会被填充或回填；
- 默认手续费为单边0.05%（万分之5）；默认模式为均衡仓位，即目标波动率20%、仓位调整带10%；
- 内置数据快照覆盖至2026-08-25，实际覆盖日期以页面“标的数据覆盖”表为准；
- 更新数据后，应重新检查日期覆盖、价格有效性、复权连续性和绩效结果。

本项目用于策略研究和展示，不构成投资建议。
