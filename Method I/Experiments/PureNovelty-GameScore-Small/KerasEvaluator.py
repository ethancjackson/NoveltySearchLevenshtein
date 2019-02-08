class KerasEvaluator:

    def __init__(self, conv_module):
        self.conv_module = conv_module

    def eval(self, input):
        # Evaluate conv module
        return self.conv_module.predict(input)
