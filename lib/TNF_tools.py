
import argparse
import os.path as osp
import lib.layers as layers
import six
from lib.train_misc import set_cnf_options

SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]


def parse_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser("FC_NF")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dims", type=str, default="8,32,32,8")
    parser.add_argument(
        "--aug_hidden_dims",
        type=str,
        default=None,
        help="The hiddden dimension of the odenet taking care of augmented dimensions",
    )
    parser.add_argument(
        "--aug_dim",
        type=int,
        default=0,
        help="The dimension along which input is augmented. 0 for 1-d input",
    )
    parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
    parser.add_argument(
        "--num_blocks", type=int, default=1, help="Number of stacked CNFs."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="ode_rnn",
        choices=["ode_rnn", "rnn", "np", "attentive_np"],
    )

    parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--layer_type",
        type=str,
        default="ignore",
        choices=[
            "ignore",
            "concat",
            "concat_v2",
            "squash",
            "concatsquash",
            "concatcoord",
            "hyper",
            "blend",
        ],
    )
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="approximate",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="softplus",
        choices=["tanh", "relu", "softplus", "elu", "swish"],
    )
    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--aug_size", type=int, default=1, help="size of time")
    parser.add_argument(
        "--latent_size", type=int, default=10, help="size of latent dimension"
    )
    parser.add_argument(
        "--rec_size", type=int, default=20, help="size of the recognition network"
    )
    parser.add_argument(
        "--rec_layers",
        type=int,
        default=1,
        help="number of layers in recognition network(ODE)",
    )
    parser.add_argument(
        "-u",
        "--units",
        type=int,
        default=100,
        help="Number of units per layer in encoder ODE func",
    )
    parser.add_argument(
        "-g",
        "--gru-units",
        type=int,
        default=100,
        help="Number of units per layer in each of GRU update networks in encoder",
    )
    parser.add_argument(
        "-n",
        "--num_iwae_samples",
        type=int,
        default=3,
        help="Number of samples to train IWAE encoder",
    )
    parser.add_argument(
        "--niwae_test", type=int, default=25, help="Numver of IWAE samples during test"
    )
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--time_length", type=float, default=1.0)
    parser.add_argument("--train_T", type=eval, default=True)
    parser.add_argument("--aug_mapping", action="store_true")
    parser.add_argument(
        "--activation", type=str, default="exp", choices=["exp", "softplus", "identity"]
    )

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument(
        "--amsgrad", action="store_true", help="use amsgrad for adam optimizer"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value for sgd optimizer"
    )

    parser.add_argument("--decoder_frequency", type=int, default=3)
    parser.add_argument("--aggressive", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--autoencode", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=True, choices=[True, False])
    parser.add_argument("--multiscale", type=eval, default=False, choices=[True, False])
    parser.add_argument("--parallel", type=eval, default=False, choices=[True, False])

    # Regularizations
    parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
    parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
    parser.add_argument(
        "--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2"
    )
    parser.add_argument(
        "--JFrobint", type=float, default=None, help="int_t ||df/dx||_F"
    )
    parser.add_argument(
        "--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
    )
    parser.add_argument(
        "--JoffdiagFrobint",
        type=float,
        default=None,
        help="int_t ||df/dx - df_i/dx_i||_F",
    )

    parser.add_argument(
        "--time_penalty", type=float, default=0, help="Regularization on the end_time."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)",
    )

    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="ctfp")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--no_tb_log", action="store_true", help="Do not use tensorboard logging"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="The split of dataset to evaluate the model on",
    )
    args = parser.parse_args()
    args.save = osp.join("experiments", args.save)

    args.effective_shape = args.input_size
    return args


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def build_TNF(args, dims, regularization_fns):
    

    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    def build_cnf():
        diffeq = layers.AugODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            effective_shape=args.effective_shape,
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            aug_dim=args.aug_dim,
            aug_mapping=args.aug_mapping,
            aug_hidden_dims=args.aug_hidden_dims,
        )
        odefunc = layers.AugODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
            effective_shape=args.effective_shape,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
        )
        return cnf
    
    
    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
            for _ in range(args.num_blocks)
        ]
        bn_chain = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
        ]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)
    set_cnf_options(args, model)
    
        
    return model

