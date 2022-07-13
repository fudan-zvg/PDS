import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import sys



config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results.")
flags.DEFINE_float("speed_up",1,"The times of speedup.")
flags.DEFINE_float("alpha",5,"The parameter alpha.")
flags.DEFINE_string("freq_mask_path",None,"The path of the frequency mask.")
flags.DEFINE_string("space_mask_path",None,"The path of the spatial mask.")
flags.mark_flags_as_required(["workdir", "config"])

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main(args):
    run_lib.evaluate(FLAGS.config,
                     FLAGS.workdir,
                     FLAGS.eval_folder,
                     FLAGS.speed_up,
                     FLAGS.freq_mask_path,
                     FLAGS.space_mask_path,
                     FLAGS.alpha)


if __name__ == "__main__":
    app.run(main)