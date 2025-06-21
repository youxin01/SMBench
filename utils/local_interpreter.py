from .base_interpreter import BaseCodeInterpreter
from .notebook_serializer import NotebookSerializer
import jupyter_client
from .log_util import logger
import os
from .response import (
    CoderMessage,
    ErrorModel,
    OutputItem,
    ResultModel,
    StdErrModel,
    StdOutModel,
    SystemMessage,
)

class LocalCodeInterpreter(BaseCodeInterpreter):
    def __init__(
        self,
        task_id: str,
        work_dir: str,
        notebook_serializer: NotebookSerializer,
    ):
        super().__init__(task_id, work_dir, notebook_serializer)
        self.km, self.kc = None, None
        self.interrupt_signal = False


    def initialize(self):
        # 本地内核一般不需异步上传文件，直接切换目录即可
        # km: 内核管理器，负责关闭和启动内核。
        # kc: 内核客户端，负责执行代码和获取输出。
        logger.info("初始化本地内核")
        self.km, self.kc = jupyter_client.manager.start_new_kernel(
            kernel_name="python3"
        )
        self._pre_execute_code()

    def _pre_execute_code(self):
        init_code = (
            f"import os\n"
            f"work_dir = r'{self.work_dir}'\n"
            f"os.makedirs(work_dir, exist_ok=True)\n"
            f"os.chdir(work_dir)\n"
            f"print('当前工作目录:', os.getcwd())\n"
            f"import matplotlib.pyplot as plt\n"
            f"import matplotlib as mpl\n"
            # 更完整的中文字体配置
            f"plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'DejaVu Sans', 'sans-serif']\n"
            f"plt.rcParams['axes.unicode_minus'] = False\n"
            f"plt.rcParams['font.family'] = 'sans-serif'\n"
            f"mpl.rcParams['font.size'] = 12\n"
            f"mpl.rcParams['axes.labelsize'] = 12\n"
            f"mpl.rcParams['xtick.labelsize'] = 10\n"
            f"mpl.rcParams['ytick.labelsize'] = 10\n"
            # 设置DPI以获得更清晰的显示
        )
        self.execute_code_(init_code)

    def execute_code(self, code: str) -> tuple[str, bool, str]:
        logger.info(f"执行代码: {code}")
        #  添加代码到notebook
        self.notebook_serializer.add_code_cell_to_notebook(code)

        text_to_gpt: list[str] = []
        content_to_display: list[OutputItem] | None = []
        error_occurred: bool = False
        error_message: str = ""

        # await redis_manager.publish_message(
        #     self.task_id,
        #     SystemMessage(content="开始执行代码"),
        # )
        # 执行 Python 代码
        logger.info("开始在本地执行代码...")
        # 返回所有的结果
        execution = self.execute_code_(code)
        logger.info("代码执行完成，开始处理结果...")

        # await redis_manager.publish_message(
        #     self.task_id,
        #     SystemMessage(content="代码执行完成"),
        # )

        for mark, out_str in execution:
            # 处理文本信息
            if mark in ("stdout", "execute_result_text", "display_text"):
                text_to_gpt.append(self._truncate_text(f"[{mark}]\n{out_str}")) #截断保留重要休息
                #  添加text到notebook
                content_to_display.append(
                    ResultModel(type="result", format="text", msg=out_str)
                )
                self.notebook_serializer.add_code_cell_output_to_notebook(out_str)
            # 处理图像信息
            elif mark in (
                "execute_result_png",
                "execute_result_jpeg",
                "display_png",
                "display_jpeg",
            ):
                # TODO: 视觉模型解释图像
                text_to_gpt.append(f"[{mark} 图片已生成，内容为 base64，未展示]")

                #  添加image到notebook
                if "png" in mark:
                    self.notebook_serializer.add_image_to_notebook(out_str, "image/png")
                    content_to_display.append(
                        ResultModel(type="result", format="png", msg=out_str)
                    )
                else:
                    self.notebook_serializer.add_image_to_notebook(
                        out_str, "image/jpeg"
                    )
                    content_to_display.append(
                        ResultModel(type="result", format="jpeg", msg=out_str)
                    )

            elif mark == "error":
                error_occurred = True
                error_message = self.delete_color_control_char(out_str)
                error_message = self._truncate_text(error_message)
                logger.error(f"执行错误: {error_message}")
                text_to_gpt.append(error_message)
                #  添加error到notebook
                self.notebook_serializer.add_code_cell_error_to_notebook(out_str)
                content_to_display.append(StdErrModel(msg=out_str))

        logger.info(f"text_to_gpt: {text_to_gpt}")
        combined_text = "\n".join(text_to_gpt)

        return (
            combined_text,
            error_occurred,
            error_message,
        )

    def execute_code_(self, code) -> list[tuple[str, str]]:
        msg_id = self.kc.execute(code)
        logger.info(f"执行代码: {code}")
        # 获得所有的返回值
        msg_list = []
        while True:
            try:
                iopub_msg = self.kc.get_iopub_msg(timeout=1)
                msg_list.append(iopub_msg)
                if (
                    iopub_msg["msg_type"] == "status"
                    and iopub_msg["content"].get("execution_state") == "idle"
                ):
                    break
            except:
                if self.interrupt_signal:
                    self.km.interrupt_kernel()
                    self.interrupt_signal = False
                continue
        # 处理返回的信息
        all_output: list[tuple[str, str]] = []
        for iopub_msg in msg_list:
            if iopub_msg["msg_type"] == "stream":
                if iopub_msg["content"].get("name") == "stdout":
                    output = iopub_msg["content"]["text"]
                    all_output.append(("stdout", output))
            elif iopub_msg["msg_type"] == "execute_result":
                if "data" in iopub_msg["content"]:
                    if "text/plain" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/plain"]
                        all_output.append(("execute_result_text", output))
                    if "text/html" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/html"]
                        all_output.append(("execute_result_html", output))
                    if "image/png" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/png"]
                        all_output.append(("execute_result_png", output))
                    if "image/jpeg" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/jpeg"]
                        all_output.append(("execute_result_jpeg", output))
            elif iopub_msg["msg_type"] == "display_data":
                if "data" in iopub_msg["content"]:
                    if "text/plain" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/plain"]
                        all_output.append(("display_text", output))
                    if "text/html" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["text/html"]
                        all_output.append(("display_html", output))
                    if "image/png" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/png"]
                        all_output.append(("display_png", output))
                    if "image/jpeg" in iopub_msg["content"]["data"]:
                        output = iopub_msg["content"]["data"]["image/jpeg"]
                        all_output.append(("display_jpeg", output))
            elif iopub_msg["msg_type"] == "error":
                # TODO: 正确返回格式
                if "traceback" in iopub_msg["content"]:
                    output = "\n".join(iopub_msg["content"]["traceback"])
                    cleaned_output = self.delete_color_control_char(output)
                    all_output.append(("error", cleaned_output))
        return all_output

    def get_created_images(self, section: str) -> list[str]:
        """获取当前 section 创建的图片列表"""
        files = os.listdir(self.work_dir)
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                self.add_section(section)
                self.section_output[section]["images"].append(file)

            self.created_images = list(
                set(self.section_output[section]["images"]) - set(self.created_images)
            )
            logger.info(f"{section}-获取创建的图片列表: {self.created_images}")
            return self.created_images

    def cleanup(self):
        # 关闭内核
        self.kc.shutdown()
        logger.info("关闭内核")
        self.km.shutdown_kernel()

    def send_interrupt_signal(self):
        self.interrupt_signal = True

    def restart_jupyter_kernel(self):
        self.kernel_client.shutdown()
        self.kernel_manager, self.kernel_client = (
            jupyter_client.manager.start_new_kernel(kernel_name="python3")
        )
        self.interrupt_signal = False
        self._create_work_dir()
