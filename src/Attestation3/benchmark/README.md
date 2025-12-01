# Benchmark для выбора алгоритма диаризации.

Датасет для теста был урезан до 2 файлов т. к был создан специально для публикации и не содержит конфиденциальной информации. Разметка выполнялась в LM Studio.

Для запуска бенчмарка необходимо скачать onnx модели и распаковать из в корень папки.

В результате были получены следующие метрики:
|name|micro_ari|macro_ari|
|----|---------|---------|
|resemblyzer + silero_vad|0,525|0,507|
|pyannote-segmentation-3-0 + 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced|0,174|0,113|
|pyannote-segmentation-3-0 + voxblink2_samresnet34_ft|0,167|0,113|
|pyannote-segmentation-3-0 + voxblink2_samresnet100_ft|0,421|0,404|
|pyannote-segmentation-3-0 + voxceleb_gemini_dfresnet114_LM|0,276|0,179|
|pyannote-segmentation-3-0 + wespeaker_en_voxceleb_resnet293_LM|0,007|0,003|
|pyannote-segmentation-3-0 + nemo_en_titanet_large|0,175|0,113|