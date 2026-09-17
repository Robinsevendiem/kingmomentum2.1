# Tushare 本地快照

该目录用于保存按原项目流程生成的 Tushare 前复权日线 Parquet 文件，与 `data/` 中的 PandaData 快照完全分开。

生成方式：在项目根目录配置本机环境变量 `TUSHARE_TOKEN` 后，运行项目提供的 Tushare 全量下载流程。生成并通过校验后，将本目录下的 `*.parquet` 文件随项目提交到 GitHub。

不要在本目录或仓库中保存 Token、原始认证文件或其他密钥。
