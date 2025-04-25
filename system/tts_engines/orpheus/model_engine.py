"""
Version: 1.0
Last Updated: 2024

Implementation Notes:
- All core system variables must be maintained
- All functions prefixed with "DONT CHANGE" must remain unmodified
- Code additions should be placed between the marked sections in each function
- Debug messages use self.print_message() with appropriate message types
- All functions must implement self.debug_func_entry() for trace logging

Note: Text between "↑↑↑ Keep everything above this line ↑↑↑" and "↓↓↓ Keep everything below this line ↓↓↓"
markers **TYPICALLY** must remain unchanged as it contains critical system integration code.

Note: You can add new functions, just DONT remove the functions that are already there, even if they
are doing nothing as `tts_server.py` will still look for their existance and fail if they are missing.
"""

########################################
# Default imports # Do not change this #
########################################
import os
import gc
import sys
import glob
import json
import time
import inspect
import torch
import logging
from pathlib import Path
from fastapi import (HTTPException)
logging.disable(logging.WARNING)

# Confguration file management for confignew.json
try:
    from system.tts_engines.config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # TGWUI import
except ImportError:
    from config import AlltalkConfig, AlltalkTTSEnginesConfig, AlltalkNewEnginesConfig # Standalone import

def initialize_configs():
    """Initialize all configuration instances"""
    config = AlltalkConfig.get_instance()
    tts_engines_config = AlltalkTTSEnginesConfig.get_instance()
    new_engines_config = AlltalkNewEnginesConfig.get_instance()
    return config, tts_engines_config, new_engines_config

# Load in the central config management
config, tts_engines_config, new_engines_config = initialize_configs()


######################################################
# Get Pytorch & Python versions # Do not change this #
######################################################
pytorch_version = torch.__version__
cuda_version = torch.version.cuda
major, minor, micro = sys.version_info[:3]
python_version = f"{major}.{minor}.{micro}"

############################################
# DeepSpeed imports # POSSIBLY change this #
############################################
"""
If the new TTS engine you are importing doesnt support DeepSpeed, then
you can simply change `model_supports_deepspeed_true_or_false` to `False`
This will be much faster when starting up the engine. This is seperate
from what is stored in your `model_settings.json` file.
"""
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ↓↓↓ MODIFY THIS LINE ↓↓↓
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

model_supports_deepspeed_true_or_false = False

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ↑↑↑ MODIFY THIS LINE ↑↑↑
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

if model_supports_deepspeed_true_or_false:
    try:
        import deepspeed
        deepspeed_available = True
    except ImportError:
        deepspeed_available = False
        pass

#######################################################
# TTS Engine-Specific Imports and Setup # Change this #
#######################################################
"""
This section is for TTS engine specific imports and global variable setup.

Guidelines:
1. Import all required modules for your TTS engine
2. Handle import errors with appropriate error messages
3. Set up any global variables or configurations needed by your engine
4. Use try/except blocks to gracefully handle missing dependencies
5. Ensure all error messages follow the AllTalk format for print messages.
   At this stage the `def print_messages` is not available, so please use
   standard `print("[AllTalk ENG] some message here")` messages.

Example structure:

try:
    # Import your TTS engine's required modules
    from your_tts_engine import required_modules
except ModuleNotFoundError:
    # Handle missing dependencies with helpful error messages
    print("Missing required modules. Please install...")
    raise

"""
try:
    # Assuming orpheus_tts is installed via pip install orpheus-speech
    from orpheus_tts import OrpheusModel
    import wave
    import io
except ModuleNotFoundError:
    brand = "[AllTalk ENG]"
    print(f"{brand} \033[91mError\033[0m Could not find the Orpheus TTS modules.")
    print(f"{brand} \033[91mError\033[0m Please install the orpheus-speech library: pip install orpheus-speech")
    raise


#########################################################
# Class setup # Change the relevant functions as needed #
#########################################################
class tts_class:
    """
    TTS Engine Implementation Class

    This class provides the interface between `tts_server.py` and whatever TTS engine you install.
    It handles model loading, voice management, and TTS generation in both streaming
    and non-streaming modes. Streaming will only be supported if the underlying TTS engine
    actually supports streaming.

    Key Responsibilities:
    1. Model Management:
       - Loading/unloading models
       - Managing model state between CPU and GPU
       - Handling DeepSpeed integration

    2. Voice Management:
       - Managing voice samples or model files

    3. TTS Generation:
       - Converting text to speech
       - Supporting streaming output (If the engine supports it)
       - Managing generation parameters

    4. System Integration:
       - Implementing standard AllTalk interfaces
       - Managing engine state and configuration
       - Handling resource allocation
    """

    ###############################################
    # Central print function # Do not change this #
    ###############################################
    def print_message(self, message, message_type="standard", component="ENG"):
        """
        Centralized print function for messages. Use this for print output to console.
        As this is the model Engine, all `component` printouts are set to ENG as default.

        Args:
            message (str): The message to print
            message_type (str): Type of message (standard/warning/error/debug_*/debug)
            component (str): Component identifier (TTS/ENG/GEN/API/etc.)

        Example Use:
            self.print_message("This is a standard print out mesage to a user)
            self.print_message("This is a debug_tts message, message_type="debug_tts")
            self.print_message("This is an error message to a user, message_type="error")
            self.print_message("This is an warning message to a user, message_type="warning")

        Debug Types:
            debug_func: Tracks function entry
            debug_tts: Enable TTS process debugging
            debug_tts_variables: Enable variable state debugging

        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        # ANSI color codes
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        prefix = f"[{config.branding}{component}] "
        if message_type.startswith("debug_"):
            debug_flag = getattr(config.debugging, message_type, False)
            if not debug_flag:
                return
            if message_type == "debug_func" and "Function entry:" in message:
                message_parts = message.split("Function entry:", 1)
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} Function entry:{GREEN}{message_parts[1]}{RESET} in model_engine")
            else:
                print(f"{prefix}{BLUE}Debug{RESET} {YELLOW}{message_type}{RESET} {message}")
        elif message_type == "debug":
            print(f"{prefix}{BLUE}Debug{RESET} {message}")
        elif message_type == "warning":
            print(f"{prefix}{YELLOW}Warning{RESET} {message}")
        elif message_type == "error":
            print(f"{prefix}{RED}Error{RESET} {message}")
        else:
            print(f"{prefix}{message}")

    def debug_func_entry(self):
        """Log function entry if debug_func is enabled."""
        if config.debugging.debug_func:
            current_func = inspect.currentframe().f_back.f_code.co_name
            self.print_message(f"Function entry: {current_func}", "debug_func")

    #############################################
    # Script initalisation # Do not change this #
    #############################################
    def __init__(self):
        """
        Initialize the TTS engine instance.

        WARNING: This class requires specific variables to interface with AllTalk's main system (tts_server.py).
        Do not remove or rename any of the predefined variables as they are required for proper system integration.

        Required System Interface Variables:

        1. Core System Variables (DO NOT MODIFY):
           Base Configuration:
           - self.this_dir: Engine directory path (where this script is located)
           - self.main_dir: AllTalk root directory
           - self.device: Processing device ("cuda" or "cpu")
           - self.cuda_is_available: Whether GPU/CUDA is available

           State Tracking:
           - self.tts_generating_lock: Prevents concurrent generation requests
           - self.tts_stop_generation: Signals generation stop request
           - self.tts_narrator_generatingtts: Tracks narrator mode for optimization
           - self.model: Active TTS model instance
           - self.is_tts_model_loaded: Whether a model is currently loaded
           - self.current_model_loaded: Name of currently loaded model
           - self.available_models: List of models found by scan_models_folder
           - self.setup_has_run: Tracks if setup() has completed

        2. Engine Configuration Variables (DO NOT MODIFY):
           - self.engines_available: List of all available TTS engines
           - self.engine_loaded: Currently selected TTS engine
           - self.selected_model: Currently selected model name

        3. Model Settings (SET VIA model_settings.json):
           Capability Flags:
           - self.audio_format: Output audio format (wav, mp3, etc.)
           - self.deepspeed_capable: DeepSpeed acceleration support
           - self.generationspeed_capable: Speed adjustment support
           - self.languages_capable: Multi-language support
           - self.lowvram_capable: Low VRAM mode support
           - self.multimodel_capable: Multiple model support
           - self.repetitionpenalty_capable: Repetition penalty support
           - self.streaming_capable: Audio streaming support
           - self.temperature_capable: Temperature adjustment support
           - self.multivoice_capable: Multiple voice support
           - self.pitch_capable: Pitch adjustment support

           Engine Settings:
           - self.def_character_voice: Default character voice
           - self.def_narrator_voice: Default narrator voice
           - self.deepspeed_enabled: DeepSpeed status
           - self.engine_installed: Engine installation status
           - self.generationspeed_set: Current speed setting
           - self.lowvram_enabled: Low VRAM mode status
           - self.repetitionpenalty_set: Current repetition penalty
           - self.temperature_set: Current temperature setting
           - self.pitch_set: Current pitch setting

           OpenAI Voice Mappings:
           - self.openai_alloy: Alloy voice mapping
           - self.openai_echo: Echo voice mapping
           - self.openai_fable: Fable voice mapping
           - self.openai_nova: Nova voice mapping
           - self.openai_onyx: Onyx voice mapping
           - self.openai_shimmer: Shimmer voice mapping

        Integration Requirements:
        - All variables must be present even if unused by your engine
        - Capability flags should accurately reflect engine features
        - Settings should have sensible defaults even if not used
        - OpenAI mappings should be set even if not supporting OpenAI compatibility

        Note: Variables marked (DO NOT MODIFY) are critical system integration points.
        Other variables should be configured through their respective JSON files or
        you can add new central variables in the section provided down below.
        """
        # DO NOT MODIFY - Sets up the base variables required for any tts engine #
        self.this_dir = Path(__file__).parent.resolve()
        self.main_dir = Path(__file__).parent.parent.parent.parent.resolve()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_is_available = torch.cuda.is_available()
        self.tts_generating_lock = False
        self.tts_stop_generation = False
        self.tts_narrator_generatingtts = False
        self.model = None
        self.is_tts_model_loaded = False
        self.current_model_loaded = None
        self.available_models = None
        self.setup_has_run = False
        self.engines_available = tts_engines_config.get_engine_names_available()
        self.engine_loaded = tts_engines_config.engine_loaded
        self.selected_model = tts_engines_config.selected_model

        # DO NOT MODIFY - Load in the current TTS Engines model_settings.json file
        settings_path = os.path.join(self.this_dir, "model_settings.json")
        try:
            with open(settings_path, 'r') as f:
                model_settings_file = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
             self.print_message(f"Error loading settings from {settings_path}: {e}. Using default settings.", message_type="error")
             model_settings_file = {
                "model_details": {"manufacturer_name": "Orpheus", "manufacturer_website": "https://github.com/canopylabs/Orpheus-TTS"},
                "model_capabilties": {
                    "audio_format": "wav",
                    "deepspeed_capable": False,
                    "generationspeed_capable": False,
                    "languages_capable": False, # Assuming single language based on original code
                    "lowvram_capable": False, # Orpheus model is initialized directly and doesn't seem to have a clear VRAM handling
                    "multimodel_capable": True, # Orpheus supports different model names
                    "repetitionpenalty_capable": True,
                    "streaming_capable": True, # Orpheus generate_speech yield tokens
                    "temperature_capable": True,
                    "multivoice_capable": True, # Orpheus supports multiple voices
                    "pitch_capable": False # Assuming no pitch control based on original code
                },
                "settings": {
                    "def_character_voice": "tara", # Default from original code
                    "def_narrator_voice": "tara", # Default from original code
                    "deepspeed_enabled": False,
                    "engine_installed": True,
                    "generationspeed_set": 1.0,
                    "lowvram_enabled": False,
                    "repetitionpenalty_set": 1.2, # Default from original code
                    "temperature_set": 0.7, # Default from original code
                    "pitch_set": 0.0 # Default value
                },
                "openai_voices": { # Default mappings, can be adjusted in model_settings.json
                    "alloy": "tara",
                    "echo": "tara",
                    "fable": "tara",
                    "nova": "tara",
                    "onyx": "tara",
                    "shimmer": "tara"
                }
            }

        # DO NOT MODIFY - Model details from model_settings.json
        self.manufacturer_name = model_settings_file["model_details"]["manufacturer_name"]
        self.manufacturer_website = model_settings_file["model_details"]["manufacturer_website"]

        # DO NOT MODIFY - Model capabilities from model_settings.json
        self.audio_format = model_settings_file["model_capabilties"]["audio_format"]
        self.deepspeed_capable = model_settings_file["model_capabilties"]["deepspeed_capable"]
        self.deepspeed_available = 'deepspeed' in globals() and model_supports_deepspeed_true_or_false
        self.generationspeed_capable = model_settings_file["model_capabilties"]["generationspeed_capable"]
        self.languages_capable = model_settings_file["model_capabilties"]["languages_capable"]
        self.lowvram_capable = model_settings_file["model_capabilties"]["lowvram_capable"]
        self.multimodel_capable = model_settings_file["model_capabilties"]["multimodel_capable"]
        self.repetitionpenalty_capable = model_settings_file["model_capabilties"]["repetitionpenalty_capable"]
        self.streaming_capable = model_settings_file["model_capabilties"]["streaming_capable"]
        self.temperature_capable = model_settings_file["model_capabilties"]["temperature_capable"]
        self.multivoice_capable = model_settings_file["model_capabilties"]["multivoice_capable"]
        self.pitch_capable = model_settings_file["model_capabilties"]["pitch_capable"]

        # DO NOT MODIFY - Engine settings from model_settings.json
        self.def_character_voice = model_settings_file["settings"]["def_character_voice"]
        self.def_narrator_voice = model_settings_file["settings"]["def_narrator_voice"]
        self.deepspeed_enabled = model_settings_file["settings"]["deepspeed_enabled"]
        self.engine_installed = model_settings_file["settings"]["engine_installed"]
        self.generationspeed_set = model_settings_file["settings"]["generationspeed_set"]
        self.lowvram_enabled = model_settings_file["settings"]["lowvram_enabled"]
        self.lowvram_enabled = False if not torch.cuda.is_available() else self.lowvram_enabled
        self.repetitionpenalty_set = model_settings_file["settings"]["repetitionpenalty_set"]
        self.temperature_set = model_settings_file["settings"]["temperature_set"]
        self.pitch_set = model_settings_file["settings"]["pitch_set"]

        # DO NOT MODIFY - OpenAI voice mappings from model_settings.json
        self.openai_alloy = model_settings_file["openai_voices"]["alloy"]
        self.openai_echo = model_settings_file["openai_voices"]["echo"]
        self.openai_fable = model_settings_file["openai_voices"]["fable"]
        self.openai_nova = model_settings_file["openai_voices"]["nova"]
        self.openai_onyx = model_settings_file["openai_voices"]["onyx"]
        self.openai_shimmer = model_settings_file["openai_voices"]["shimmer"]

        """
        Below is the name of the folder that will be created-used under `/models/{folder}`
        And is further used by the base functions of this script. Change the name stored in
        `self.model_folder_name` to the model folder name you will be using. Ensure to use
        the correct CAPS/Non-CAPS spelling as Linux OS is CAPS specific on folder names.
        """
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ MODIFY THIS LINE ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        self.model_folder_name = "Orpheus"

        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ MODIFY THIS LINE ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Add your own central `self.myvariable` variables in here if needed for your engine ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        """
        If you need globally accessable variables of your own for your own purposes, you can put
        them in here as self.myvariable = "whatever".

        """
        # The Orpheus model is initialized with model_name, store it here
        self.orpheus_model_name = model_settings_file["settings"].get("model_name", "unsloth/orpheus-3b-0.1-ft")

        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Add your own central `self.myvariable` variables in here if needed for your engine ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # DO NOT MODIFY - log the function call to this function
        self.debug_func_entry()

    #####################################################
    # Printout engine loading bits # Do not change this #
    #####################################################
    def printout_versions(self):
        """
        Print Python, DeepSpeed, Pytorch and CUDA version on start-up.

        WARNING: This is a core system function. Do not modify its implementation
        as it provides standardized version reporting across all engines.
        """
        self.debug_func_entry()
        if not model_supports_deepspeed_true_or_false:
            self.print_message(f"\033[92mDeepSpeed version :\033[93m Not supported on {self.model_folder_name}\033[0m", message_type="standard")
        else:
            if deepspeed_available:
                self.print_message("\033[92mDeepSpeed version :\033[93m " + str(deepspeed.__version__) + "\033[0m", message_type="standard")
            else:
                self.print_message("\033[92mDeepSpeed version :\033[93m Not available\033[0m", message_type="standard")
        self.print_message(f"\033[92mPython Version    :\033[93m {python_version}\033[0m", message_type="standard")
        self.print_message(f"\033[92mPyTorch Version   :\033[93m {pytorch_version}\033[0m", message_type="standard")
        if cuda_version is None:
            self.print_message("\033[92mCUDA Version      :\033[91m Not available\033[0m", message_type="standard")
        else:
            self.print_message(f"\033[92mCUDA Version      :\033[93m {cuda_version}\033[0m", message_type="standard")

        self.print_message("", message_type="standard")
        return

    ################################################################
    # Handle low VRAM change between CUDA/CPU # Do not change this #
    ################################################################
    async def handle_lowvram_change(self):
        """
        Manage model location between CPU and GPU memory for low VRAM operation.

        This function handles the movement of models between CPU and GPU memory
        to support systems with limited VRAM. It's called automatically during
        generation when low VRAM mode is enabled.

        Operation:
        1. Checks CUDA availability
        2. Moves model between devices based on current location:
           - GPU (cuda) -> CPU
           - CPU -> GPU (cuda)
        3. Manages CUDA cache to optimize memory usage

        States Affected:
        - self.device: Updated to reflect current processing device
        - self.model.device: Model's current memory location

        Requirements:
        - CUDA must be available for GPU operations
        - Model must be loaded (self.model is not None)
        - lowvram_enabled must be `True` in the `model_settings.json` file

        Note: This function is only called when self.lowvram_enabled is True
        meaning the engine does or doesnt support the call, hence if its not
        True, then this function would never be called anyway, so doesnt need
        changing.
        """
        self.debug_func_entry()

        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")

        if torch.cuda.is_available():
            if self.device == "cuda":
                self.print_message("Moving model to CPU", message_type="debug_tts")
                self.device = "cpu"
                if hasattr(self.model, 'to'): # Check if model supports .to()
                    self.model.to(self.device)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                self.device = "cuda"
                self.print_message("Moving model to GPU", message_type="debug_tts")
                if hasattr(self.model, 'to'): # Check if model supports .to()
                    self.model.to(self.device)
                gc.collect()


    ################################################
    # Handle DeepSpeed change # Do not change this #
    ################################################
    async def handle_deepspeed_change(self, value):
        """
        Handle enabling/disabling of DeepSpeed acceleration.

        This function manages the process of reloading the model with or without
        DeepSpeed acceleration. DeepSpeed can significantly improve performance on
        supported hardware.

        Args:
            value (bool): True to enable DeepSpeed, False to disable

        Operation:
        1. Unloads current model
        2. Updates DeepSpeed settings
        3. Reloads model with new configuration

        States Affected:
        - self.deepspeed_enabled: Updated to reflect new state
        - self.model: Reloaded with new configuration

        Returns:
            bool: The new DeepSpeed state (same as input value)

        Note: DeepSpeed must be installed and available in the system for
        this functionality to work. `deepspeed_capable` must be set `True`
        in the `model_settings.json` file
        """
        self.debug_func_entry()
        # Initial validation
        if not self.is_tts_model_loaded:
            self.print_message("No model is currently loaded. Please select a model to load.", message_type="error")
            raise HTTPException(status_code=400, detail="No model is currently loaded. Please select a model to load.")

        if value:
            self.print_message("\033[93mDeepSpeed Activating\033[0m", message_type="standard")
            await self.unload_model()
            self.deepspeed_enabled = True
            await self.setup()
        else:
            self.print_message("\033[93mDeepSpeed De-Activating\033[0m", message_type="standard")
            self.deepspeed_enabled = False
            await self.unload_model()
            await self.setup()
        return value


    ################################################################
    # Unload models from VRAM/RAM if possible # Do not change this #
    ################################################################
    async def unload_model(self):
        """
        Unload the current model and free associated resources.

        This function handles the cleanup of model resources, including:
        1. Setting model loaded flag to False
        2. Deleting the model instance
        3. Clearing CUDA cache if available

        Operation:
        1. Updates model loading status
        2. Logs unloading process if a model is loaded
        3. Removes model from memory
        4. Cleans up CUDA cache if using GPU

        States Affected:
        - self.is_tts_model_loaded: Set to False
        - self.model: Set to None after unloading
        """
        self.debug_func_entry()

        self.is_tts_model_loaded = False
        if not self.current_model_loaded == None:
            self.print_message("Unloading model", message_type="debug_tts")
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    ############################################################
    # On start-up, perform these actions # Change as necessary #
    ############################################################
    async def setup(self):
        """
        Initialize the TTS engine and load initial model configuration.

        This function is called during system startup and handles:
        1. Version information display
        2. Model scanning and availability check
        3. Initial model loading if specified

        The setup sequence ensures:
        - Proper version reporting
        - Model availability verification
        - Graceful handling of missing models
        - Correct initial model loading state

        States Set:
        - self.available_models: Updated with found models
        - self.current_model_loaded: Set to loaded model name or None
        - self.setup_has_run: Set True when complete

        Returns:
            None

        Note: Custom initialization code should be placed between the marked sections.
        """
        self.debug_func_entry()
        self.print_message("Initializing TTS engine", message_type="debug_tts")
        self.printout_versions()
        self.available_models = self.scan_models_folder()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Orpheus model initialization
        try:
            self.print_message(f"Initializing Orpheus model: {self.orpheus_model_name}", message_type="standard")
            self.model = OrpheusModel(model_name=self.orpheus_model_name)
            self.print_message("Orpheus model loaded successfully.", message_type="standard")
            self.is_tts_model_loaded = True
            self.current_model_loaded = self.orpheus_model_name # Orpheus uses a single model name but different voices
        except Exception as e:
            self.print_message(f"Error loading Orpheus model: {e}", message_type="error")
            self.model = None # Set model to None on failure
            self.is_tts_model_loaded = False
            self.current_model_loaded = "Failed to Load"


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if not self.is_tts_model_loaded:
            self.print_message(f"Selected model '{self.selected_model}' not found in models folder.", message_type="error")
            self.print_message(f"Please download a model or select a different model file.", message_type="error")

        self.setup_has_run = True


    ###########################################################################
    # Scan your models folder for models OR voice files # Change as necessary #
    ###########################################################################
    def scan_models_folder(self):
        """
        Scan for available TTS models in the models directory.

        This function searches the models directory for valid TTS model installations.
        Each model must contain all required files to be considered valid.

        Required Files for Each Model:
        - Whatever your TTS engine needs/supports

        Operation:
        1. Scans the models/{folder} directory
        2. Checks each subfolder for required files
        3. Registers valid models

        States Affected:
        - self.available_models: Updated with found models

        Returns:
            dict: Dictionary of available models in format:
                 {model_identifier: engine_type}

        Note: If no valid models are found, returns {"No Models Available": "TTS Engine Name"}
        """
        self.debug_func_entry()

        # For Orpheus, the "model" is the initialized engine itself,
        # and the "voices" are the selectable options.
        # We will represent the single initialized model as the available one.
        self.available_models = {self.orpheus_model_name: self.model_folder_name}


        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        # ↑↑↑ Keep everything above this line ↑↑↑
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # Orpheus does not load models from local files in the same way as XTTS or Piper.
        # The available models conceptually refers to the initialized OrpheusModel instance.

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        if not self.available_models:
            self.available_models = {"No Models Found": self.model_folder_name}
            self.print_message(f"No models found for {self.model_folder_name}", message_type="warning")
            self.print_message("Please ensure the selected Orpheus model name is valid if issues arise.", message_type="warning")

        return self.available_models



    ################################################################
    # Scan your voice folder for voice files # Change as necessary #
    ################################################################
    def voices_file_list(self):
        """
        Scan and compile a list of available voices

        This function scans multiple directories to find voice models/samples in different formats:
        1. Individual WAV/MP3 etc files in the main voices directory

        Directory Structure:
        - voices/: Individual audio files

        Returns:
            list: Available voices with appropriate prefixes:
                 - Standard WAV: filename.wav

        Note: Returns ["No Voices Found"] if no valid voices are detected
        """
        self.debug_func_entry()

        # For Orpheus, the voices are predefined or callable from the installed model.
        # This list should reflect the voices the Orpheus model supports.
        # This information is often available through the Orpheus library itself
        # or could be hardcoded based on the model in use.
        # Assuming 'tara' and 'brian' are available voices for the default model.
        # If Orpheus provides a way to list available voices, use that instead.

        try:
            voices = [] # An empy variable for the list of voices to be put into.
            directory = self.main_dir / "voices" # Base directory that voices are stored in.
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # Add known voices for the default Orpheus model
            voices.append("tara")
            voices.append("brian")
            # If using a different model, dynamically add its voices if possible.
            # Placeholder: In a real scenario, you might query the OrpheusModel instance:
            # if self.model:
            #    voices.extend(self.model.list_available_voices()) # Hypothetical function

            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # ↓↓↓ Keep everything below this line ↓↓↓
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            # Sort voices by type alphabetically - Not applicable for Orpheus simple voice list
            # voices.sort(key=lambda x: (x.startswith("voiceset:"), x.startswith("latent:"), x))
            if not voices:
                return ["No Voices Found"]
            return voices

        except Exception as e:
            self.print_message(f"Error scanning for voices: {str(e)}", message_type="error")
            return ["No Voices Found"]

    ############################################
    # Load in your model # Change as necessary #
    ############################################
    async def load_model(self, model_name):
        """
        Load a model using the your TTS API interface.

        Args:
            model_name (str): Name of the model to load

        Operation:
        1. Validates model availability
        2. Constructs model and config paths
        3. Initializes model using TTS API
        4. Moves model to appropriate device (CPU/GPU)

        States Affected:
        - self.model: Updated with loaded model
        - self.is_tts_model_loaded: Set to True on success

        Returns:
            The loaded model instance

        Raises:
            HTTPException: If no models are available to load
        """
        self.debug_func_entry()
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return

        # For Orpheus, loading a model means ensuring the OrpheusModel instance is initialized.
        # The actual model data fetching is handled by the Orpheus library when the instance is created.
        # We should check if the requested model_name matches the name used to initialize the OrpheusModel.
        if model_name == self.orpheus_model_name and self.model is not None:
             self.print_message(f"Orpheus model '{model_name}' is already loaded.", message_type="standard")
             self.is_tts_model_loaded = True
             return self.model
        
        # If the requested model_name is different, we'd need to re-initialize OrpheusModel,
        # assuming the Orpheus library supports this after initial creation.
        # The original Orpheus code initializes it once in __init__.
        # A more robust implementation might re-initialize here with the new model_name
        # after unloading the old one. For now, stick to the original behavior of
        # assuming a single model initialized at startup matching self.orpheus_model_name.
        # If a different model_name is requested, we might need to raise an exception or handle it differently.
        # For now, assume the requested model_name is always self.orpheus_model_name.

        self.print_message(f"Attempting to load Orpheus model: {model_name}. Note: Orpheus typically loads the model specified in settings at startup.", message_type="warning")
        # The actual loading happens in self.setup(), which is called at initialization and handle_tts_method_change.
        # Since handle_tts_method_change calls unload_model and then setup, if a different model_name was passed
        # and Orpheus supported dynamic loading, setup would re-initialize with the new model_name from settings.
        # So, ensure settings.json is updated with the desired model_name before calling this.
        # Given the current structure, model_name parameter here might be slightly redundant vs. settings.

        self.is_tts_model_loaded = True # Assume loaded if we reached here and self.model is not None from setup
        if self.model is None:
             self.print_message("Orpheus model could not be loaded. Check settings and logs.", message_type="error")
             self.is_tts_model_loaded = False
             raise HTTPException(status_code=500, detail="Orpheus model failed to load.")

        return self.model


    async def handle_tts_method_change(self, tts_method):
        """
        Handle switching between different TTS models/voices.

        This function manages actual model loading process.

        Args:
            tts_method (str): Format "type - modelname" where type is either

        Operation:
        1. Validates model availability
        2. Unloads current model if any
        3. Parses method string to determine loader type
        4. Calls appropriate model loader
        5. Updates current model tracking

        States Affected:
        - self.current_model_loaded: Updated to new model identifier
        - self.model: Updated with newly loaded model

        Returns:
            bool: True if model loaded successfully, False otherwise

        Timing:
            Records and reports model loading time
        """
        self.debug_func_entry()

        # Track loading time
        generate_start_time = time.time()

        # Validate model availability
        if "No Models Available" in self.available_models:
            self.print_message("No models for this TTS engine were found to load", message_type="error")
            return False

        # Unload current model (only needed if Orpheus supported dynamic unloading/reloading)
        # await self.unload_model()

        # For Orpheus, the 'tts_method' seems to indicate the chosen model name
        # if self.multimodel_capable. We should ensure the OrpheusModel
        # instance (initialized in setup) is ready.
        # The original Orpheus code doesn't have explicit model switching after initialization.
        # If a user selects a different model in the UI, we would need to restart the engine
        # or implement dynamic model loading/unloading in the OrpheusModel itself if the library supports it.
        # For now, we'll simply check if the currently loaded model matches the request.

        requested_model_name = tts_method # Assuming tts_method is the model name

        if self.current_model_loaded == requested_model_name and self.is_tts_model_loaded:
            self.print_message(f"Requested model '{requested_model_name}' is already loaded.", message_type="standard")
            # No unloading or reloading needed
            self.selected_model = requested_model_name # Update selected_model in instance
            # You might also want to update the central tts_engines.json here if this is meant to persist
            tts_engines_config.set_selected_model(requested_model_name)
            tts_engines_config.save_config()

        elif requested_model_name in self.available_models:
             self.print_message(f"Attempting to switch to Orpheus model: {requested_model_name}. Note: Orpheus typically loads the model specified in settings at startup, engine restart may be needed.", message_type="warning")
             # In a real scenario where Orpheus supported dynamic switching, you would unload and then load.
             # await self.unload_model()
             # await self.load_model(requested_model_name) # This call assumes load_model can handle a new model_name after init
             # For now, reflect the selected model but warn the user.
             self.current_model_loaded = "Switch Pending/Partial" # Indicate a state change was attempted but might not be fully reflected without restart
             self.is_tts_model_loaded = False # Indicate potential need for restart
             self.selected_model = requested_model_name # Update selected_model in instance
             tts_engines_config.set_selected_model(requested_model_name)
             tts_engines_config.save_config()

        else:
            self.print_message(f"Requested Orpheus model '{requested_model_name}' not found in available models.", message_type="error")
            return False


        # Report loading time
        generate_end_time = time.time()
        generate_elapsed_time = generate_end_time - generate_start_time
        self.print_message(f"\033[94mModel Loadtime: \033[93m{generate_elapsed_time:.2f}\033[94m seconds\033[0m")
        # Return True if the model is loaded, even if it's the one from setup
        return self.is_tts_model_loaded # Return True if model is loaded and ready


    async def generate_tts(self, text, voice, language, temperature, repetition_penalty, speed, pitch, output_file, streaming):
        """
        Generate speech from text using the TTS model.

        This core function handles all TTS generation, supporting both streaming and
        non-streaming output, multiple voice input types,
        and various generation parameters.

        Args:
            text (str): Text to convert to speech
            voice (str): Voice identifier (WAV/MP3 etc. file, voiceset, or latent)
            language (str): Target language code
            temperature (float): Generation temperature (0.0-1.0)
            repetition_penalty (float): Penalty for repetitive generation
            speed (float): Speech speed multiplier
            pitch (float): Voice pitch adjustment 
            output_file (str): Path for output audio file
            streaming (bool): Whether to stream audio chunks

        Returns:
            For streaming=True: Generator yielding audio chunks
            For streaming=False: None (saves to output_file)

        States Used:
            - self.model: Active TTS model
            - self.device: Current processing device
            - self.lowvram_enabled: Low VRAM mode status
            - self.current_model_loaded: Current model type
        """
        self.debug_func_entry()

        # Initial validation
        if voice == "No Voices Found":
            self.print_message("No voices found to generate TTS.", message_type="error")
            raise HTTPException(status_code=400, detail="No voices found to generate TTS.")

        if not self.is_tts_model_loaded:
            self.print_message("No TTS model loaded", message_type="error")
            raise HTTPException(status_code=400, detail="You currently have no TTS model loaded.")


        # Lock generation and track start time
        self.tts_generating_lock = True
        self.print_message("Starting TTS generation process", message_type="debug_tts")
        self.print_message(f"Generation parameters: temperature={temperature}, repetition_penalty={repetition_penalty}, speed={speed}, pitch={pitch}, streaming={streaming}",
                        message_type="debug_tts_variables") # Added repetition_penalty, speed, pitch for completeness

        # Handle low VRAM mode if needed - Orpheus doesn't support this based on initial analysis
        # if self.lowvram_enabled and self.device == "cpu":
        #     self.print_message("Low VRAM mode: Moving model to GPU", message_type="debug_tts")
        #     await self.handle_lowvram_change()

        generate_start_time = time.time()

        try:
            # Voice input processing - Orpheus uses voice name directly
            self.print_message(f"Using voice: {voice}", message_type="debug_tts")

            # Apply speed and pitch settings if capable (Orpheus assumed not capable based on initial analysis)
            # if self.generationspeed_capable:
            #     # Apply speed setting if your engine supports it
            #     pass
            # if self.pitch_capable:
            #     # Apply pitch setting if your engine supports it
            #     pass

            # Get generation arguments, falling back to instance settings if None
            # and then to default values if instance settings are not set directly
            temp_set = temperature if temperature is not None else self.temperature_set if self.temperature_set is not None else 0.7
            rep_penalty_set = repetition_penalty if repetition_penalty is not None else self.repetitionpenalty_set if self.repetitionpenalty_set is not None else 1.2
            # Speed and pitch are not supported by Orpheus based on initial code, use default/ignored values
            speed_set = speed if speed is not None else self.generationspeed_set if self.generationspeed_set is not None else 1.0
            pitch_set = pitch if pitch is not None else self.pitch_set if self.pitch_set is not None else 0.0


            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            # ↑↑↑ Keep everything above this line ↑↑↑
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

            # BUILD ANY SETTINGS NEEED FOR TTS HERE
            # DO ANY LOGIC TESTS E.G. CHECK IF THE AUDIO FILE/MODEL EXISTS

            # ALWAYS CHECK IF STREAMING FIRST THEN DO OTHER METHODS
            if streaming and self.streaming_capable:
                 self.print_message("Starting streaming generation", message_type="debug_tts")
                 # PUT YOUR TTS ENGINE STREAMING GENERATION LOGIC IN HERE
                 syn_tokens = self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    temperature=temp_set,
                    top_p=rep_penalty_set, # Orpheus uses top_p, the template uses repetition_penalty in parameters. Map repetition_penalty to top_p.
                    repetition_penalty=rep_penalty_set, # Orpheus also takes repetition_penalty
                 )
                 # Orpheus yields bytes directly, yield them as chunks
                 for audio_chunk in syn_tokens:
                     if isinstance(audio_chunk, bytes):
                         yield audio_chunk
                     else:
                         self.print_message(f"Warning: Received non-bytes chunk type during streaming: {type(audio_chunk)}", message_type="warning")

            else:
                self.print_message("Starting non-streaming generation", message_type="debug_tts")

                # PUT YOUR TTS ENGINE STANDARD GENERATION LOGIC IN HERE
                # Generate speech tokens (streaming output)
                syn_tokens = self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    temperature=temp_set,
                    top_p=rep_penalty_set, # Map repetition_penalty to top_p.
                    repetition_penalty=rep_penalty_set, # Orpheus also takes repetition_penalty
                )

                # Write streaming audio to a byte buffer in WAV format for non-streaming output
                byte_buffer = io.BytesIO()

                with wave.open(byte_buffer, "wb") as wf:
                    # Orpheus outputs 16-bit mono audio at 24000 Hz
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 2 bytes for 16-bit
                    wf.setframerate(24000)

                    total_frames = 0
                    for audio_chunk in syn_tokens:
                        if isinstance(audio_chunk, bytes):
                             frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                             total_frames += frame_count
                             wf.writeframes(audio_chunk)
                        else:
                            self.print_message(f"Warning: Received non-bytes chunk type during non-streaming: {type(audio_chunk)}", message_type="warning")


                # Get the byte content of the WAV file and save to output_file
                audio_data = byte_buffer.getvalue()
                byte_buffer.close()

                try:
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    self.print_message(f"Saved audio to: {output_file}", message_type="debug_tts")
                except Exception as e:
                    self.print_message(f"Error saving audio to {output_file}: {e}", message_type="error")
                    raise HTTPException(status_code=500, detail=f"Error saving audio file: {e}") from e


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # ↓↓↓ Keep everything below this line ↓↓↓
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        finally:
            # Generation complete
            generate_end_time = time.time()
            generate_elapsed_time = generate_end_time - generate_start_time

            # Standard output message (not debug)
            self.print_message(
                f"\033[94mTTS Generate: \033[93m{generate_elapsed_time:.2f} seconds. \033[94mLowVRAM: \033[33m{self.lowvram_enabled} \033[94mDeepSpeed: \033[33m{self.deepspeed_enabled}\033[0m",
                message_type="standard"
            )

            # Handle low VRAM cleanup - Orpheus assumed not capable
            # if self.lowvram_enabled and self.device == "cuda" and not self.tts_narrator_generatingtts:
            #     self.print_message("Low VRAM mode: Moving model back to CPU", message_type="debug_tts")
            #     await self.handle_lowvram_change()

            self.tts_generating_lock = False