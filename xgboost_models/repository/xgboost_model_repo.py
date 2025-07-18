from ..models import XGBoostModels


class XGBoostModelRepository:
    @staticmethod
    def get_all_models():
        return XGBoostModels.objects.all()

    @staticmethod
    def get_model_by_id(pk):
        return XGBoostModels.objects.filter(pk=pk).first()

    @staticmethod
    def save_model_info(model_info):
        model = XGBoostModels.objects.create(**model_info)
        return model

    @staticmethod
    def update_model_info(pk, updated_info):
        XGBoostModels.objects.filter(pk=pk).update(**updated_info)
        return XGBoostModels.objects.get(pk=pk)

    @staticmethod
    def delete_model_by_id(pk):
        XGBoostModels.objects.filter(pk=pk).delete()
