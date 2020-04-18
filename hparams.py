from text import symbols

class HParams(object):
    def __init__(self):
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs=500
        self.iters_per_checkpoint=1000
        self.seed=1234
        self.dynamic_loss_scaling=True
        self.fp16_run=True
        self.distributed_run=True
        self.dist_backend="nccl"
        self.dist_url="tcp://localhost:54321"
        self.cudnn_enabled=True
        self.cudnn_benchmark=False
        self.ignore_layers=['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk=False
        self.training_files='filelists/librispeech_gst_with_id_train.txt'
        self.validation_files='filelists/librispeech_gst_with_id_eval.txt'
        self.text_cleaners=['english_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value=32768.0
        self.sampling_rate=16000
        self.filter_length=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mel_channels=80
        self.mel_fmin=0.0
        self.mel_fmax=8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols=len(symbols)
        self.symbols_embedding_dim=512

        # Encoder parameters
        self.encoder_kernel_size=5
        self.encoder_n_convolutions=3
        self.encoder_embedding_dim=512

        # Decoder parameters
        self.n_frames_per_step=2
        self.decoder_rnn_dim=1024
        self.prenet_dim=256
        self.max_decoder_steps=1000
        self.gate_threshold=0.5
        self.p_attention_dropout=0.1
        self.p_decoder_dropout=0.1

        # Attention parameters
        self.attention_rnn_dim=1024
        self.attention_dim=128

        # Location Layer parameters
        self.attention_location_n_filters=32
        self.attention_location_kernel_size=31

        # Mel-post processing network parameters
        self.postnet_embedding_dim=512
        self.postnet_kernel_size=5
        self.postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate=False
        self.learning_rate=1e-3
        self.weight_decay=1e-6
        self.grad_clip_thresh=1.0
        self.batch_size=32
        self.mask_padding=True  # set model's padded outputs to padded values

        #######################
        # GST Hyperparameters #
        #######################      
        self.E = 256
        self.use_gst = True
        self.ref_enc_filters=[16, 16, 32, 32, 64, 64]
        self.ref_enc_size = [3, 3]
        self.ref_enc_strides = [2, 2]
        self.ref_enc_pad = [1, 1]
        self.ref_enc_gru_size = self.E // 2
        self.num_heads = 8
        self.token_num = 10
        
        #####################################
        # Auxiliary Embedding HyperParameters #
        #####################################       
        self.auxiliary_embedding_num = 2 # 1172 for clean 100+360 speaker id, 2 for sex
        self.auxiliary_embedding_dim = 16 # 256 for speaker id, 16 for sex
        

def create_hparams():
    return HParams()
