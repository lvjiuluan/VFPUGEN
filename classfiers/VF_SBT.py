from classfiers.VF_BASE import VF_BASE_CLF, VF_BASE_REG
from utils.FateUtils import *
from consts.Constants import *
from utils.pklUtils import *
import subprocess
import sys
from utils.Logger import Logger


class VF_SBT():
    def __init__(self, config):
        """
        初始化 VF_SBT 实例，加载并更新配置。
        """
        try:
            sbt_config = load_config(SBT_CONFIGS_PATH)
            sbt_config.update(config)
            self.config = sbt_config
            log_level = config.get('log_level', 'ERROR')
            self.logger = Logger.create_new_logger(log_level)
            self.result = None
            self.objective = None
        except Exception as e:
            self.logger.info(f"初始化时发生错误: {e}")
            raise

    def fit(self, XA, XB, y, objective, num_class):
        """
        训练 SBT 模型。
        """
        try:
            self.config['objective'] = objective
            self.config['num_class'] = num_class
            self.objective = SbtObjective(objective)
            save_config(self.config, SBT_CONFIGS_PATH)
            A_df, B_df = fate_construct_df(XA, XB, y)
            save_host_guest_dataframes(A_df, B_df, A_host_train_path, B_guest_train_path)
            self.logger.info("VF_SBT训练结束")
        except Exception as e:
            self.logger.info(f"训练时发生错误: {e}")
            raise

    def predict(self, XA, XB):
        """
        进行预测并返回分类结果。
        """
        return self._execute_prediction(XA, XB, return_proba=False)

    def predict_proba(self, XA, XB):
        """
        进行预测并返回预测概率,多维数组，从0到C，每一列表示一类的概率
        """
        return self._execute_prediction(XA, XB, return_proba=True)

    def _execute_prediction(self, XA, XB, return_proba=False):
        """
        执行预测的私有方法，减少代码重复。

        :param XA: 主数据集
        :param XB: 客数据集
        :param return_proba: 是否返回预测概率
        :return: 预测结果或预测概率
        """

        try:
            A_df, B_df = fate_construct_df(XA, XB)
            save_host_guest_dataframes(A_df, B_df, A_host_test_path, B_guest_test_path)
            self.result = execute_sbt_command(self.config, self.logger)

            # 提取结果
            self.guest_result = self.result.get('guest', {})
            self.host_result = self.result.get('host', {})

            self.guest_model_dict = self.guest_result.get('model_dict', {})
            self.host_model_dict = self.host_result.get('model_dict', {})
            self.pred_df = self.guest_result.get('pred_df', None)

            if self.pred_df is None:
                raise ValueError("预测数据框 `pred_df` 缺失。")

            self.predict_score = self.pred_df.predict_score
            self.predict_result = self.pred_df.predict_result
            self.y_pred = self.predict_result

            self.y_proba = None
            if self.objective in {SbtObjective.BINARY_BCE, SbtObjective.MULTI_CE}:
                self.y_proba = parse_probability_details(self.pred_df.predict_detail)

            return self.y_proba if return_proba else self.predict_result

        except Exception as e:
            self.logger.info(f"预测时发生错误: {e}")
            raise


class VF_SBT_CLF(VF_SBT, VF_BASE_CLF):
    """
    分类器类，继承自 VF_SBT_BASE 和 VF_BASE_CLF。
    """

    def fit(self, XA, XB, y):
        """
        调用基类的 fit 方法。
        """
        objective, num_class = determine_task_type(y)
        super().fit(XA, XB, y, objective, num_class)

    def predict(self, XA, XB):
        """
        调用基类的 _execute_prediction 方法。
        """
        return self._execute_prediction(XA, XB, return_proba=False)

    def predict_proba(self, XA, XB):
        """
        调用基类的 _execute_prediction 方法。
        """
        return self._execute_prediction(XA, XB, return_proba=True)


class VF_SBT_REG(VF_SBT, VF_BASE_REG):
    """
    回归器类，继承自 VF_SBT_BASE 和 VF_BASE_REG。
    """

    def fit(self, XA, XB, y):
        """
        调用基类的 fit 方法。
        """
        objective, num_class = SbtObjective.REGRESSION_L2, None
        super().fit(XA, XB, y, objective, num_class)

    def predict(self, XA, XB):
        """
        调用基类的 _execute_prediction 方法。
        """
        return self._execute_prediction(XA, XB, return_proba=False)
