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

    def model_dump_to_callable(self, kallable, *args, **kwargs):
        """This function is intended to be used to safely pass the model's settings as K/V arguments to a callable.
        It takes a callable and additional arguments and keyword arguments.
        It returns the result of calling the callable, passing the K/V arguments as returned by the calling
        the model_dump function, passing the additional arguments and keyword arguments.
        Before passing the result of the model_dump function to the callable, None and empty values are removed
        and secrets are replaced with their values.

        This is a safer alternative to using func_kwarg_model_dump function directly.
        """
        return kallable(self.func_kwarg_model_dump(*args, **kwargs))




