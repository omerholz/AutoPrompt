from pydantic import BaseModel

class ConfigBase(BaseModel):
    def func_kwarg_model_dump(self, *args, **kwargs):
        """Warning: This function reveals secrets. Never print or log its output.
        This function is intended to be used to pass the model's settings as K/V arguments to a function.
        It returns a dictionary of the model's settings, including secrets, and excluding None or empty values.
        """
        return {
            k: v if not hasattr(v, 'get_secret_value') else v.get_secret_value()
            for k, v in self.model_dump(*args, **kwargs).items()
            if k and v
        }


