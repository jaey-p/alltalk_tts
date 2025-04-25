import os
import json
import requests
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from .help_content import AllTalkHelpContent
this_dir = Path(__file__).parent.resolve()                         # Sets up self.this_dir as a variable for the folder THIS script is running in.
main_dir = Path(__file__).parent.parent.parent.parent.resolve()    # Sets up self.main_dir as a variable for the folder AllTalk is running in

##########################################################################
# REQUIRED CHANGE                                                        #
# Populate the voices list, using the method specific to your TTS engine #
##########################################################################
# This function is responsible for populating the list of available voices for your TTS engine.
# You need to modify this function to use the appropriate method for your engine to retrieve the voice list.
#
# The current implementation lists all the WAV files in a "voices" directory, which may not be suitable for your engine.
# You should replace the `xxxx_voices_file_list` function name to match your engine name. For example, if your engine
# is named "mytts", the function should be named `mytts_voices_file_list`.
#
# You will also neef to update the code with your own implementation that retrieves the voice list according to your
# engine's specific requirements. Typically this is the same code as will be in your model_engine.py file.
#
# For example, if your engine has a dedicated API or configuration file for managing voices, you should modify this
# function to interact with that API or read from that configuration file.
#
# After making the necessary changes, this function should return a list of available voices that can be used
# in your TTS engine's settings page.

# This function needs access to the Orpheus engine instance to call its voices_file_list method.
# Assuming the engine instance is passed to the main settings page function.
# Let's define a placeholder function shape that expects the engine instance.
def orpheus_voices_file_list():
    """Gathers a list of available voice files in the voices directory."""
    voice_dir = main_dir / "voices"
    voice_list = [f for f in os.listdir(voice_dir) if f.endswith('.wav')]
    if not voice_list:
        return ["No voices found in ./voices/"]
    return voice_list


######################################################
# REQUIRED CHANGE                                    #
# Imports and saves the TTS engine-specific settings #
######################################################
# This function is responsible for importing and saving the settings specific to your TTS engine.
# You need to make the following change:
#
# 1. Change the name of the function `xxxx_model_update_settings` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_model_update_settings`.
#
# After making this change, the function will load the model settings from a JSON file, update the settings and voice
# dictionaries with the values provided as arguments, and save the updated settings back to the JSON file.
#
# You do not need to modify the function's logic or any other part of the code.

def orpheus_model_update_settings(model_name_gr, def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr,  alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr):
    """Updates the Orpheus model settings in model_settings.json."""
    # Load the model_config_data from the JSON file
    settings_path = os.path.join(this_dir, "model_settings.json")
    try:
        with open(settings_path, "r") as f:
            model_config_data = json.load(f)
    except Exception as e:
        return f"Error loading settings: {e}"


    # Update the settings and openai_voices dictionaries with the new values
    # Ensure we handle potential missing keys gracefully
    if "settings" not in model_config_data:
        model_config_data["settings"] = {}
    if "openai_voices" not in model_config_data:
        model_config_data["openai_voices"] = {}

    model_config_data["settings"]["model_name"] = model_name_gr # Add model_name handling
    model_config_data["settings"]["def_character_voice"] = def_character_voice_gr
    model_config_data["settings"]["def_narrator_voice"] = def_narrator_voice_gr
    model_config_data["openai_voices"]["alloy"] = alloy_gr
    model_config_data["openai_voices"]["echo"] = echo_gr
    model_config_data["openai_voices"]["fable"] = fable_gr
    model_config_data["openai_voices"]["nova"] = nova_gr
    model_config_data["openai_voices"]["onyx"] = onyx_gr
    model_config_data["openai_voices"]["shimmer"] = shimmer_gr
    # These capabilities are assumed False for Orpheus, but update if interactive in UI
    if "lowvram_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["lowvram_capable"]:
        model_config_data["settings"]["lowvram_enabled"] = lowvram_enabled_gr == "Enabled"
    if "deepspeed_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["deepspeed_capable"]:
         model_config_data["settings"]["deepspeed_enabled"] = deepspeed_enabled_gr == "Enabled"

    if "temperature_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["temperature_capable"]:
        model_config_data["settings"]["temperature_set"] = temperature_set_gr
    if "repetitionpenalty_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["repetitionpenalty_capable"]:
        model_config_data["settings"]["repetitionpenalty_set"] = repetitionpenalty_set_gr
    if "pitch_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["pitch_capable"]:
         model_config_data["settings"]["pitch_set"] = pitch_set_gr
    if "generationspeed_capable" in model_config_data.get("model_capabilties", {}).keys() and model_config_data["model_capabilties"]["generationspeed_capable"]:
        model_config_data["settings"]["generationspeed_set"] = generationspeed_set_gr


    # Save the updated model_config_data to the JSON file
    try:
        with open(settings_path, "w") as f:
            json.dump(model_config_data, f, indent=4)
    except Exception as e:
         return f"Error saving settings: {e}"


    return "Settings updated successfully! Restart engine or AllTalk for some settings to take full effect."

#######################################################
# REQUIRED CHANGE                                     #
# Sets up the engine-specific settings page in Gradio #
#######################################################
# This function sets up the Gradio interface for the settings page specific to your TTS engine.
# You need to make the following changes:
#
# 1. Change the name of the function `xxxx_model_alltalk_settings` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_model_alltalk_settings`.
#
# 2. Change the name of the `submit_button.click` function call to match the name you gave to the function
#    that imports and saves your engine's settings (the function you modified above).
#
# 3. Change the name of the `voice_list` function call to match the name of the function that lists
#    the available voices for your TTS engine.
#
# 4. Change the 'title' of the `gr.Blocks` to match your engine's name e.g. title="mytts TTS"
#
# After making these changes, this function will create and return the Gradio interface for your TTS engine's
# settings page, allowing users to configure various options and voice selections.

# This function needs the engine instance to pass to the voice listing function.
def orpheus_model_alltalk_settings(model_config_data):
    """Sets up the Gradio interface for the Orpheus settings page."""
    features_list = model_config_data.get('model_capabilties', {})

    # Get voices using the helper function
    voice_list = orpheus_voices_file_list()

    with gr.Blocks(title="Orpheus TTS", analytics_enabled=False) as app:
        with gr.Tab("Default Settings"):
            with gr.Row():
                # Add the model name input
                model_name_gr = gr.Textbox(label="Orpheus Model Name", value=model_config_data["settings"].get("model_name", ""), interactive=features_list.get("multimodel_capable", True)) # Assume interactive if multimodel_capable isn't explicitly False
                lowvram_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="Low VRAM" if features_list.get("lowvram_capable", False) else "Low VRAM N/A", value="Enabled" if model_config_data["settings"].get("lowvram_enabled", False) else "Disabled", interactive=features_list.get("lowvram_capable", False))
                deepspeed_enabled_gr = gr.Radio(choices={"Enabled": "true", "Disabled": "false"}, label="DeepSpeed Activate" if features_list.get("deepspeed_capable", False) else "DeepSpeed N/A", value="Enabled" if model_config_data["settings"].get("deepspeed_enabled", False) else "Disabled", interactive=features_list.get("deepspeed_capable", False))
                temperature_set_gr = gr.Slider(value=float(model_config_data["settings"].get("temperature_set", 0.7)), minimum=0, maximum=1, step=0.05, label="Temperature" if features_list.get("temperature_capable", False) else "Temperature N/A", interactive=features_list.get("temperature_capable", False))
                repetitionpenalty_set_gr = gr.Slider(value=float(model_config_data["settings"].get("repetitionpenalty_set", 1.2)), minimum=1, maximum=20, step=1, label="Repetition Penalty" if features_list.get("repetitionpenalty_capable", False) else "Repetition N/A", interactive=features_list.get("repetitionpenalty_capable", False))
                pitch_set_gr = gr.Slider(value=float(model_config_data["settings"].get("pitch_set", 0.0)), minimum=-10, maximum=10, step=1, label="Pitch" if features_list.get("pitch_capable", False) else "Pitch N/A", interactive=features_list.get("pitch_capable", False))
                generationspeed_set_gr = gr.Slider(value=float(model_config_data["settings"].get("generationspeed_set", 1.0)), minimum=0.25, maximum=2.00, step=0.25, label="Speed" if features_list.get("generationspeed_capable", False) else "Speed N/A", interactive=features_list.get("generationspeed_capable", False))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OpenAI Voice Mappings")
                    with gr.Group():
                        with gr.Row():
                            alloy_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("alloy", "tara"), label="Alloy", choices=voice_list, allow_custom_value=True)
                            echo_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("echo", "tara"), label="Echo", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            fable_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("fable", "tara"), label="Fable", choices=voice_list, allow_custom_value=True)
                            nova_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("nova", "tara"), label="Nova", choices=voice_list, allow_custom_value=True)
                        with gr.Row():
                            onyx_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("onyx", "tara"), label="Onyx", choices=voice_list, allow_custom_value=True)
                            shimmer_gr = gr.Dropdown(value=model_config_data["openai_voices"].get("shimmer", "tara"), label="Shimmer", choices=voice_list, allow_custom_value=True)
                with gr.Column():
                    gr.Markdown("### Default Voices")
                    with gr.Row():
                        def_character_voice_gr = gr.Dropdown(value=model_config_data["settings"].get("def_character_voice", "tara"), label="Default/Character Voice", choices=voice_list, allow_custom_value=True)
                        def_narrator_voice_gr = gr.Dropdown(value=model_config_data["settings"].get("def_narrator_voice", "tara"), label="Narrator Voice", choices=voice_list, allow_custom_value=True)
                    with gr.Group():
                        with gr.Row():
                            details_text = gr.Textbox(label="Details", show_label=False, lines=5, interactive=False, value="Configure default settings and voice mappings for the selected TTS engine. Unavailable options are grayed out based on engine capabilities. See the Help section below for detailed information about each setting.")
            with gr.Row():
                submit_button = gr.Button("Update Settings")
                output_message = gr.Textbox(label="Output Message", interactive=False, show_label=False)
            with gr.Accordion("HELP - 🔊 Understanding TTS Engine Default Settings Page", open=False):
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS, elem_classes="custom-markdown")
                with gr.Row():
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS1, elem_classes="custom-markdown")
                    gr.Markdown(AllTalkHelpContent.DEFAULT_SETTINGS2, elem_classes="custom-markdown")
            # Update the click function to include the model name input
            submit_button.click(orpheus_model_update_settings, inputs=[model_name_gr, def_character_voice_gr, def_narrator_voice_gr, lowvram_enabled_gr, deepspeed_enabled_gr, temperature_set_gr, repetitionpenalty_set_gr, pitch_set_gr, generationspeed_set_gr, alloy_gr, echo_gr, fable_gr, nova_gr, onyx_gr, shimmer_gr], outputs=output_message)

        ###########################################################################################
        # Do not change this section apart from "TTS Engine Name" value to match your engine name #
        ###########################################################################################
        with gr.Tab("Engine Information"):
            with gr.Row():
                with gr.Group():
                    gr.Textbox(label="Manufacturer Name", value=model_config_data['model_details'].get('manufacturer_name', 'N/A'), interactive=False)
                    gr.Textbox(label="Manufacturer Website/TTS Engine Support", value=model_config_data['model_details'].get('manufacturer_website', 'N/A'), interactive=False)
                    gr.Textbox(label="Engine/Model Description", value=model_config_data['model_details'].get('model_description', 'N/A'), interactive=False, lines=13)
                with gr.Column():
                    with gr.Row():
                        gr.Textbox(label="DeepSpeed Capable", value='Yes' if features_list.get('deepspeed_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Pitch Capable", value='Yes' if features_list.get('pitch_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Generation Speed Capable", value='Yes' if features_list.get('generationspeed_capable', False) else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Repetition Penalty Capable", value='Yes' if features_list.get('repetitionpenalty_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Multi Languages Capable", value='Yes' if features_list.get('languages_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Streaming Capable", value='Yes' if features_list.get('streaming_capable', False) else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Low VRAM Capable", value='Yes' if features_list.get('lowvram_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Temperature Capable", value='Yes' if features_list.get('temperature_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Multi Model Capable Engine", value='Yes' if features_list.get('multimodel_capable', False) else 'No', interactive=False)
                    with gr.Row():
                        gr.Textbox(label="Multi Voice Capable Models", value='Yes' if features_list.get('multivoice_capable', False) else 'No', interactive=False)
                        gr.Textbox(label="Default Audio output format", value=model_config_data['model_capabilties'].get('audio_format', 'N/A'), interactive=False)
                        gr.Textbox(label="TTS Engine Name", value="Orpheus", interactive=False) # Changed to Orpheus
                    with gr.Row():
                         gr.Textbox(label="Windows Support", value='Yes' if features_list.get('windows_capable', False) else 'No', interactive=False)
                         gr.Textbox(label="Linux Support", value='Yes' if features_list.get('linux_capable', False) else 'No', interactive=False)
                         gr.Textbox(label="Mac Support", value='Yes' if features_list.get('mac_capable', False) else 'No', interactive=False)
            with gr.Row():
                with gr.Accordion("HELP - 🔊 Understanding TTS Engine Capabilities", open=False):
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION, elem_classes="custom-markdown")
                    with gr.Row():
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION1, elem_classes="custom-markdown")
                        gr.Markdown(AllTalkHelpContent.ENGINE_INFORMATION2, elem_classes="custom-markdown")


        with gr.Tab("Models/Voices Download - Orpheus"): # Modified tab title for clarity
            with gr.Row():
                # For Orpheus, this section is less about downloading files to a folder
                # and more about managing the model name used by the Orpheus library.
                # We'll display the current model name and allow changing it.

                current_model_display = gr.Textbox(label="Currently Selected Orpheus Model", value=model_config_data["settings"].get("model_name", "N/A"), interactive=False)
                # We could add a dropdown of known Orpheus models from available_models.json
                # but for simplicity now, a text input allows specifying any model name.
                new_model_name_input = gr.Textbox(label="Enter New Orpheus Model Name (requires engine restart)", value="", interactive=features_list.get("multimodel_capable", True)) # Assume interactive if multimodel_capable isn't explicitly False
                update_model_button = gr.Button("Update Model Name in Settings")

            with gr.Row():
                model_update_status = gr.Textbox(label="Status", interactive=False)

            def update_orpheus_model_setting(new_model_name):
                """Updates the model_name setting in model_settings.json."""
                settings_path = os.path.join(this_dir, "model_settings.json")
                try:
                    with open(settings_path, "r") as f:
                        model_config_data = json.load(f)
                except Exception as e:
                    return f"Error loading settings: {e}"

                if "settings" not in model_config_data:
                    model_config_data["settings"] = {}

                model_config_data["settings"]["model_name"] = new_model_name

                try:
                    with open(settings_path, "w") as f:
                        json.dump(model_config_data, f, indent=4)
                except Exception as e:
                    return f"Error saving settings: {e}"

                return f"Model name updated to '{new_model_name}' in settings.json. Please restart the Orpheus engine or AllTalk for this change to take effect."


            update_model_button.click(update_orpheus_model_setting, inputs=new_model_name_input, outputs=model_update_status)

        ###################################################################################################
        # REQUIRED CHANGE                                                                                 #
        # Add any engine specific help, bugs, issues, operating system specifc requirements/setup in here #
        # Please use Markdown format, so gr.Markdown() with your markdown inside it.                      #
        ###################################################################################################
        with gr.Tab("Engine Help"):
            with gr.Row():
                # Use the Orpheus-specific help content
                gr.Markdown(AllTalkHelpContent.HELP_PAGE, elem_classes="custom-markdown")
            with gr.Row():
                # Use the Orpheus-specific help content
                gr.Markdown(AllTalkHelpContent.HELP_PAGE1, elem_classes="custom-markdown")
                gr.Markdown(AllTalkHelpContent.HELP_PAGE2, elem_classes="custom-markdown")

    return app

################################
# REQUIRED CHANGE              #
# Sets up the Gradio interface #
################################
# This function sets up the Gradio interface for your TTS engine's settings page.
# You need to change the name of the function calls to match the names you set in the functions above.
#
# Specifically, you need to update the following:
#
# 1. The name of the function `xxxx_at_gradio_settings_page` to match your engine's name.
#    For example, if your engine is named "mytts", the function should be named `mytts_at_gradio_settings_page`.
#
# 2. The name of the function call `xxxx_model_alltalk_settings(model_config_data)`.
#    This should match the name you gave to the function that sets up the engine-specific settings page in Gradio.
#    If you named that function `mytts_model_alltalk_settings`, then the call should be:
#    `mytts_model_alltalk_settings(model_config_data)`
#
# After making these changes, this function will create and return the Gradio app for your TTS engine's settings page.

# This main function will need to accept the engine instance.
def orpheus_at_gradio_settings_page(model_config_data):
    """Main function to set up the Orpheus Gradio settings page."""
    # Pass the engine instance to the settings layout function
    app = orpheus_model_alltalk_settings(model_config_data)
    return app