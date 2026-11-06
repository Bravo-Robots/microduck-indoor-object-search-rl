# Fuentes y alcance

- Robot, MJCF y adaptador de inferencia: Pollen Robotics,
  https://github.com/pollen-robotics/microduck_rl . El adaptador conserva la
  adaptación Apache-2.0 del taller local. Copia en `third_party/infer_policy.py`.
- Mallas STL oficiales: CC BY-NC-SA 4.0; conservar atribución y licencia al
  redistribuir o adaptar. Alcance exacto: `third_party/robot/LICENSE-MODELS.md`.
- `alpha_stand.onnx` y `velstand.onnx`: políticas de Pollen Robotics,
  https://huggingface.co/pollen-robotics/microduck-policies .
- `progresion_4_mejor.onnx`: entrenamiento PPO previo del taller local,
  tarea oficial `Mjlab-Velocity-Flat-MicroDuck`, checkpoint 1499. No se presenta
  como un checkpoint oficial de Pollen ni como entrenamiento nuevo de este proyecto.
- Modelo de actuador BAM: Rhoban, Apache-2.0,
  https://github.com/Rhoban/bam , commit fijado en `pyproject.toml` y `uv.lock`.
- Motor MuJoCo: https://github.com/google-deepmind/mujoco .
- Blender/bpy: https://www.blender.org/ .
- Vídeo de referencia: REC Anything, https://www.youtube.com/watch?v=ac6pspFbPsI .
  Capturas y transcripción son referencias externas locales, no assets originales.

`third_party/manifest.json` identifica los archivos copiados y sus SHA-256.
El mobiliario nuevo no deriva de las mallas del robot y se exporta por separado.
