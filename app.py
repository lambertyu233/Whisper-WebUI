import os
import argparse
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION, is_video, get_media_files
from modules.utils.audio_manager import batch_extract_audio
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger


logger = get_logger()




class App:
    def __init__(self, args):
        self.args = args
        # Check every 1 hour (3600) for cached files and delete them if older than 1 day (86400)
        self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row():
            dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value=whisper_params["model_size"],
                                   label=_("Model"), allow_custom_value=True)
            dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                  value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap()
                                  else whisper_params["lang"], label=_("Language"))
            dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt", "LRC"], value=whisper_params["file_format"], label=_("File Format"))
        with gr.Row():
            cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label=_("Translate to English?"),
                                       interactive=True)
        with gr.Row():
            cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                       label=_("Add a timestamp to the end of the filename"),
                                       interactive=True)
        with gr.Row():
            cb_filter_repetition = gr.Checkbox(value=whisper_params.get("filter_repetition", False),
                                               label=_("Filter repetitive/hallucinatory subtitle segments"),
                                               interactive=True)

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            whisper_type=self.args.whisper_type,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type)

        with gr.Accordion(_("Background Music Remover Filter"), open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion(_("Voice Detection Filter"), open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion(_("Diarization"), open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.diarizer.available_device,
                                                                    device=self.whisper_inf.diarizer.device)

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp,
            cb_filter_repetition
        )

    def launch(self):
        translation_params = self.default_params["translation"]
        deepl_params = translation_params["deepl"]
        nllb_params = translation_params["nllb"]
        uvr_params = self.default_params["bgm_separation"]

        with self.app:
            lang = gr.Radio(choices=list(self.i18n.keys()),
                            value="zh",
                            label=_("Language"), interactive=True,
                            visible=False,  # Set it by development purpose.
                            )
            with Translate(self.i18n, lang=lang):  # Add `lang = lang` here to test dynamic change of the languages.
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(MARKDOWN, elem_id="md_project")
                with gr.Tabs():
                    with gr.TabItem(_("File")):  # tab1
                        with gr.Column():
                            input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION,
                                                   visible=False)
                            with gr.Row():
                                btn_select_local_file = gr.Button("选择本地文件", variant="secondary")
                                btn_select_local_folder = gr.Button("选择本地文件夹", variant="secondary")

                            tb_local_files = gr.Textbox(label="已选择的本地文件路径 (直接读取，不复制/不上传)",
                                                        placeholder="未选择本地文件",
                                                        lines=3,
                                                        interactive=False)

                            tb_input_folder = gr.Textbox(label="输入本地文件夹路径 (可选)",
                                                         info="可选：如果您想直接转录本地文件，可以指定本地文件夹路径。如果不希望使用本地路径，请保留为空。",
                                                         visible=True,
                                                         value="")
                            cb_include_subdirectory = gr.Checkbox(label="包含子目录文件",
                                                                  info="当使用上述本地文件夹路径时，是否包含子文件夹中的所有文件。",
                                                                  visible=True,
                                                                  value=False)
                            cb_save_same_dir = gr.Checkbox(label="保存在输入文件同级目录下",
                                                           info="当使用上述本地文件夹路径时，是否除默认 outputs 目录外，也在输入文件相同目录下存一份输出结果。",
                                                           visible=True,
                                                           value=True)

                            btn_select_local_file.click(fn=self.on_select_local_files, outputs=[tb_local_files, input_file, tb_input_folder])
                            btn_select_local_folder.click(fn=self.on_select_local_folder, outputs=[tb_local_files, input_file, tb_input_folder])
                            
                            input_file.change(fn=lambda x: "" if x else gr.update(), inputs=input_file, outputs=tb_local_files)
                            tb_input_folder.change(fn=lambda x: (None, "") if x else (gr.update(), gr.update()), inputs=tb_input_folder, outputs=[input_file, tb_local_files])

                        pipeline_params, dd_file_format, cb_timestamp, cb_filter_repetition = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_extract_audio = gr.Button(_("GENERATE AUDIO FILE"), variant="secondary", interactive=False)
                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                            btn_openfolder = gr.Button('📂', scale=1)

                        extract_params = [input_file, tb_local_files, tb_input_folder, cb_include_subdirectory]
                        btn_extract_audio.click(fn=self.extract_audio_wrapper,
                                                inputs=extract_params,
                                                outputs=[tb_indicator, files_subtitles])

                        check_video_inputs = [input_file, tb_local_files, tb_input_folder, cb_include_subdirectory]
                        btn_select_local_file.click(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)
                        btn_select_local_folder.click(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)
                        input_file.change(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)
                        tb_local_files.change(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)
                        tb_input_folder.change(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)
                        cb_include_subdirectory.change(fn=self.on_check_has_video, inputs=check_video_inputs, outputs=btn_extract_audio)

                        params = [input_file, tb_local_files, tb_input_folder, cb_include_subdirectory, cb_save_same_dir,
                                  dd_file_format, cb_timestamp, cb_filter_repetition]
                        params = params + pipeline_params
                        btn_run.click(fn=self.transcribe_file_wrapper,
                                      inputs=params,
                                      outputs=[tb_indicator, files_subtitles])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Youtube")):  # tab2
                        with gr.Row():
                            tb_youtubelink = gr.Textbox(label=_("Youtube Link"))
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                img_thumbnail = gr.Image(label=_("Youtube Thumbnail"))
                            with gr.Column():
                                tb_title = gr.Label(label=_("Youtube Title"))
                                tb_description = gr.Textbox(label=_("Youtube Description"), max_lines=15)

                        pipeline_params, dd_file_format, cb_timestamp, cb_filter_repetition = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [tb_youtubelink, dd_file_format, cb_timestamp, cb_filter_repetition]

                        btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                      inputs=params + pipeline_params,
                                      outputs=[tb_indicator, files_subtitles])
                        tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                              outputs=[img_thumbnail, tb_title, tb_description])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Mic")):  # tab3
                        with gr.Row():
                            mic_input = gr.Microphone(label=_("Record with Mic"), type="filepath", interactive=True,
                                                      show_download_button=True)

                        pipeline_params, dd_file_format, cb_timestamp, cb_filter_repetition = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [mic_input, dd_file_format, cb_timestamp, cb_filter_repetition]

                        btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                      inputs=params + pipeline_params,
                                      outputs=[tb_indicator, files_subtitles])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("T2T Translation")):  # tab 4
                        with gr.Column():
                            file_subs = gr.Files(type="filepath", label=_("Upload Subtitle Files to translate here"))
                            btn_select_local_sub = gr.Button("选择本地字幕文件", variant="secondary")
                            
                            tb_local_subs = gr.Textbox(label="已选择的本地字幕文件路径 (直接读取，不复制/不上传)",
                                                       placeholder="未选择本地字幕文件",
                                                       lines=3,
                                                       interactive=False)
                            btn_select_local_sub.click(fn=self.on_select_local_subs, outputs=[tb_local_subs, file_subs])
                            file_subs.change(fn=lambda x: "" if x else gr.update(), inputs=file_subs, outputs=tb_local_subs)

                        with gr.TabItem(_("DeepL API")):  # sub tab1
                            with gr.Row():
                                tb_api_key = gr.Textbox(label=_("Your Auth Key (API KEY)"),
                                                        value=deepl_params["api_key"])
                            with gr.Row():
                                dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                             value=AUTOMATIC_DETECTION if deepl_params["source_lang"] == AUTOMATIC_DETECTION.unwrap()
                                                             else deepl_params["source_lang"],
                                                             choices=list(self.deepl_api.available_source_langs.keys()))
                                dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                             value=deepl_params["target_lang"],
                                                             choices=list(self.deepl_api.available_target_langs.keys()))
                            with gr.Row():
                                cb_is_pro = gr.Checkbox(label=_("Pro User?"), value=deepl_params["is_pro"])
                            with gr.Row():
                                cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                           label=_("Add a timestamp to the end of the filename"),
                                                           interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.translate_deepl_wrapper,
                                      inputs=[tb_api_key, file_subs, tb_local_subs, dd_source_lang, dd_target_lang,
                                              cb_is_pro, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles])

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                        with gr.TabItem(_("NLLB")):  # sub tab2
                            with gr.Row():
                                dd_model_size = gr.Dropdown(label=_("Model"), value=nllb_params["model_size"],
                                                            choices=self.nllb_inf.available_models)
                                dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                             value=nllb_params["source_lang"],
                                                             choices=self.nllb_inf.available_source_langs)
                                dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                             value=nllb_params["target_lang"],
                                                             choices=self.nllb_inf.available_target_langs)
                            with gr.Row():
                                nb_max_length = gr.Number(label=_("Max Length Per Line"), value=nllb_params["max_length"],
                                                          precision=0)
                            with gr.Row():
                                cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                           label=_("Add a timestamp to the end of the filename"),
                                                           interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)
                            with gr.Column():
                                md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                        btn_run.click(fn=self.translate_nllb_wrapper,
                                      inputs=[file_subs, tb_local_subs, dd_model_size, dd_source_lang, dd_target_lang,
                                              nb_max_length, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles])

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                    with gr.TabItem(_("BGM Separation")):
                        with gr.Column():
                            files_audio = gr.Files(type="filepath", label=_("Upload Audio Files to separate background music"))
                            btn_select_local_audio = gr.Button("选择本地音频文件", variant="secondary")
                            
                            tb_local_audios = gr.Textbox(label="已选择的本地音频文件路径 (直接读取，不复制/不上传)",
                                                         placeholder="未选择本地音频文件",
                                                         lines=3,
                                                         interactive=False)
                            btn_select_local_audio.click(fn=self.on_select_local_audio, outputs=[tb_local_audios, files_audio])
                            files_audio.change(fn=lambda x: "" if x else gr.update(), inputs=files_audio, outputs=tb_local_audios)

                        dd_uvr_device = gr.Dropdown(label=_("Device"), value=self.whisper_inf.music_separator.device,
                                                    choices=self.whisper_inf.music_separator.available_devices)
                        dd_uvr_model_size = gr.Dropdown(label=_("Model"), value=uvr_params["uvr_model_size"],
                                                        choices=self.whisper_inf.music_separator.available_models)
                        nb_uvr_segment_size = gr.Number(label=_("Segment Size"), value=uvr_params["segment_size"],
                                                        precision=0)
                        cb_uvr_save_file = gr.Checkbox(label=_("Save separated files to output"),
                                                       value=True, visible=False)
                        btn_run = gr.Button(_("SEPARATE BACKGROUND MUSIC"), variant="primary")
                        with gr.Column():
                            with gr.Row():
                                ad_instrumental = gr.Audio(label=_("Instrumental"), scale=8)
                                btn_open_instrumental_folder = gr.Button('📂', scale=1)
                            with gr.Row():
                                ad_vocals = gr.Audio(label=_("Vocals"), scale=8)
                                btn_open_vocals_folder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.separate_files_wrapper,
                                      inputs=[files_audio, tb_local_audios, dd_uvr_model_size, dd_uvr_device, nb_uvr_segment_size,
                                              cb_uvr_save_file],
                                      outputs=[ad_instrumental, ad_vocals])
                        btn_open_instrumental_folder.click(inputs=None,
                                                           outputs=None,
                                                           fn=lambda: self.open_folder(os.path.join(
                                                               self.args.output_dir, "UVR", "instrumental"
                                                           )))
                        btn_open_vocals_folder.click(inputs=None,
                                                     outputs=None,
                                                     fn=lambda: self.open_folder(os.path.join(
                                                         self.args.output_dir, "UVR", "vocals"
                                                     )))

        # Launch the app with optional gradio settings
        args = self.args
        self.app.queue(
            api_open=args.api_open
        ).launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=eval(args.allowed_paths) if args.allowed_paths else None
        )

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")

    def select_local_files(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            logger.error(f"Error importing tkinter: {e}")
            return None

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_patterns = " ".join([f"*{ext}" for ext in MEDIA_EXTENSION])
        filetypes = [
            ("媒体文件", file_patterns),
            ("所有文件", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="选择媒体文件",
            filetypes=filetypes
        )
        root.destroy()
        if not files:
            return None

        # Dynamically append selected file paths to allowed_paths
        for file in files:
            dir_path = os.path.dirname(os.path.abspath(file))
            if dir_path not in self.app.allowed_paths:
                self.app.allowed_paths.append(dir_path)

        return list(files)

    def select_local_folder(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            logger.error(f"Error importing tkinter: {e}")
            return ""

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder = filedialog.askdirectory(title="选择文件夹")
        root.destroy()
        if not folder:
            return ""

        abs_folder = os.path.abspath(folder)
        if abs_folder not in self.app.allowed_paths:
            self.app.allowed_paths.append(abs_folder)

        return folder

    def select_local_subs(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            logger.error(f"Error importing tkinter: {e}")
            return None

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        filetypes = [
            ("字幕文件", "*.srt *.vtt *.lrc *.txt"),
            ("所有文件", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="选择字幕文件",
            filetypes=filetypes
        )
        root.destroy()
        if not files:
            return None

        for file in files:
            dir_path = os.path.dirname(os.path.abspath(file))
            if dir_path not in self.app.allowed_paths:
                self.app.allowed_paths.append(dir_path)

        return list(files)

    def on_select_local_files(self):
        files = self.select_local_files()
        if not files:
            return gr.update(), gr.update(), gr.update()
        paths_str = "\n".join(files)
        return paths_str, None, ""

    def on_select_local_folder(self):
        folder = self.select_local_folder()
        if not folder:
            return gr.update(), gr.update(), gr.update()
        return "", None, folder

    def on_select_local_subs(self):
        files = self.select_local_subs()
        if not files:
            return gr.update(), gr.update()
        paths_str = "\n".join(files)
        return paths_str, None

    def on_select_local_audio(self):
        files = self.select_local_files()
        if not files:
            return gr.update(), gr.update()
        paths_str = "\n".join(files)
        return paths_str, None

    def on_check_has_video(self, input_file, tb_local_files, tb_input_folder, cb_include_subdirectory):
        if tb_input_folder and os.path.isdir(tb_input_folder):
            media_files = get_media_files(tb_input_folder, include_sub_directory=cb_include_subdirectory)
            has_video = any(is_video(f) for f in media_files)
            return gr.update(interactive=has_video)

        if tb_local_files:
            files = [line.strip() for line in tb_local_files.split('\n') if line.strip()]
            has_video = any(is_video(f) for f in files)
            return gr.update(interactive=has_video)

        if input_file:
            if isinstance(input_file, list):
                has_video = any(is_video(f if isinstance(f, str) else getattr(f, 'name', str(f))) for f in input_file)
            else:
                has_video = is_video(input_file if isinstance(input_file, str) else getattr(input_file, 'name', str(input_file)))
            return gr.update(interactive=has_video)

        return gr.update(interactive=False)

    def extract_audio_wrapper(self,
                              input_file,
                              tb_local_files,
                              tb_input_folder,
                              cb_include_subdirectory,
                              progress=gr.Progress()):
        files_to_extract = None
        if not tb_input_folder:
            if tb_local_files:
                files_to_extract = [line.strip() for line in tb_local_files.split('\n') if line.strip()]
            else:
                files_to_extract = input_file

        return batch_extract_audio(
            files=files_to_extract,
            input_folder_path=tb_input_folder,
            include_subdirectory=cb_include_subdirectory,
            progress=progress
        )

    def transcribe_file_wrapper(self,
                                input_file,
                                tb_local_files,
                                tb_input_folder,
                                cb_include_subdirectory,
                                cb_save_same_dir,
                                file_format,
                                add_timestamp,
                                filter_repetition,
                                *pipeline_params,
                                progress=gr.Progress()):
        files_to_transcribe = None
        if not tb_input_folder:
            if tb_local_files:
                files_to_transcribe = [line.strip() for line in tb_local_files.split('\n') if line.strip()]
            else:
                files_to_transcribe = input_file

        return self.whisper_inf.transcribe_file(
            files_to_transcribe,
            tb_input_folder,
            cb_include_subdirectory,
            cb_save_same_dir,
            file_format,
            add_timestamp,
            filter_repetition,
            progress,
            *pipeline_params
        )

    def translate_deepl_wrapper(self,
                                auth_key,
                                file_subs,
                                tb_local_subs,
                                source_lang,
                                target_lang,
                                is_pro=False,
                                add_timestamp=True,
                                progress=gr.Progress()):
        files_to_translate = None
        if tb_local_subs:
            files_to_translate = [line.strip() for line in tb_local_subs.split('\n') if line.strip()]
        else:
            files_to_translate = file_subs

        return self.deepl_api.translate_deepl(
            auth_key,
            files_to_translate,
            source_lang,
            target_lang,
            is_pro,
            add_timestamp,
            progress
        )

    def translate_nllb_wrapper(self,
                               file_subs,
                               tb_local_subs,
                               model_size,
                               src_lang,
                               tgt_lang,
                               max_length=200,
                               add_timestamp=True,
                               progress=gr.Progress()):
        files_to_translate = None
        if tb_local_subs:
            files_to_translate = [line.strip() for line in tb_local_subs.split('\n') if line.strip()]
        else:
            files_to_translate = file_subs

        return self.nllb_inf.translate_file(
            files_to_translate,
            model_size,
            src_lang,
            tgt_lang,
            max_length,
            add_timestamp,
            progress
        )

    def separate_files_wrapper(self,
                               files_audio,
                               tb_local_audios,
                               model_name,
                               device=None,
                               segment_size=256,
                               save_file=True,
                               progress=gr.Progress()):
        files_to_separate = None
        if tb_local_audios:
            files_to_separate = [line.strip() for line in tb_local_audios.split('\n') if line.strip()]
        else:
            files_to_separate = files_audio

        return self.whisper_inf.music_separator.separate_files(
            files_to_separate,
            model_name,
            device,
            segment_size,
            save_file,
            progress
        )

parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.FASTER_WHISPER.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR,
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=INSANELY_FAST_WHISPER_MODELS_DIR,
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
if __name__ == "__main__":
    _args = parser.parse_args()
    app = App(args=_args)
    app.launch()
