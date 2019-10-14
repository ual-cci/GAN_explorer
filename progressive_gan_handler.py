class ProgressiveGAN_Handler(object):
    """
    Handles a trained Progressive GAN model, loads from
    """

    def __init__(self, model_path):
        print(model_path)

    def _create_model(self, model_path):
        print(model_path)

    def report(self):
        print("typical input is ...")

    def example_input(self):
        print("example input is ...")
        example_input = None
        return example_input

    def infer(self, input):
        output = None
        return output

    def set_mode(self):
        print("This model has no modes to set ...")


pg_path = "aerials512vectors1024px_snapshot-010200.pkl"
pg_handler = ProgressiveGAN_Handler()

pg_handler.report()
example_input = pg_handler.example_input()

example_output = pg_handler.infer(example_input)

print("example_output:", example_output)