# gold_dl：黄金期货 1 分钟数据的可复用代码
#   config：读取 config.yaml，提供项目根目录和各数据路径
#   data_pipeline：tf.data 输入流水线（训练、验证、测试相同的步骤）和屏蔽空标签的损失函数
#   mbp1：分块读取每月的 MBP-1 原始文件、有效盘口判断
#   timeutil：北京时间 / UTC / 美东时间转换、CME 交易日
#   indicators：ATR、安全除法
#   labels：三重障碍标签（make_labels.py 和 make_labels_with_preroll_check.py 调用）
#   plotting：报告图的颜色、字体和 matplotlib 样式
__version__ = "0.1.0"
