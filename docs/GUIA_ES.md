# Guía de uso

El proyecto está preparado como repositorio de simulación: busca tres clases de
objetos con un supervisor RL y ejecuta la marcha en el cuerpo físico de MuJoCo.
La validación en un MicroDuck real sigue pendiente.

## Arranque

Desde la raíz del proyecto:

```bash
uv sync --locked
bash run.sh doctor
bash run.sh rehearse --viewer --no-video
```

Para una máquina sin GPU, el entrenamiento corto funciona con `--device cpu`.
El renderizado puede usar OSMesa: instala `libosmesa6` y ejecuta
`MUJOCO_GL=osmesa bash run.sh preview`. El visor necesita escritorio gráfico.

Para buscar la caja azul y guardar vídeo e informe:

```bash
bash run.sh rehearse --goal 1 --seed 3 \
  --video outputs/caja-azul.mp4 --output outputs/caja-azul.json
```

Objetivos: `0` pelota roja, `1` caja azul y `2` cilindro amarillo. `--seed` cambia
posiciones, mobiliario e inicio. Una misión termina por hallazgo, caída,
confirmación falsa, tiempo agotado o cierre del visor; el informe lo distingue.

## Entrenar

```bash
bash run.sh train --device cpu --steps 1536 --envs 8 --output outputs/prueba
bash run.sh train --steps 12000000 --envs 256 --output outputs/runs/entrenamiento
bash run.sh rehearse --checkpoint outputs/runs/entrenamiento/best.pt --no-video
```

El entrenamiento se hace en un modelo rápido de búsqueda, no en las articulaciones
completas. Después se verifica con servos y contactos en MuJoCo. La capa de freno
por profundidad se aplica por igual al RL y al controlador de comparación.

## Archivos

- [Blender editable](../assets/workshop_search.blend).
- [STL originales](../assets/stl), en metros; escalar ×1000 para milímetros.
- [Resultados actuales](../outputs/release/RESULTS.json).
- [Contrato del modelo](../outputs/release/MODEL_CARD.md).
- [Histórico del prototipo](../benchmarks/v0.1).

El código propio usa Apache-2.0. Las mallas oficiales y derivados que las
contienen conservan CC BY-NC-SA 4.0; ver [atribución](../THIRD_PARTY.md).

## Comprobar antes de subir

```bash
uv run ruff check duck_search scripts tests
uv run ruff format --check duck_search scripts tests
uv run python -m unittest discover -s tests -v
uv run python scripts/check_repo.py
uv run python scripts/package_release.py
```

El ZIP resultante contiene los archivos públicos y sus hashes. Los entornos,
referencias de vídeo y pruebas temporales quedan fuera. La CI se ejecuta en ambos repositorios de GitHub. Incluye una búsqueda en un
pasillo despejado para verificar percepción, avance con motores y confirmación.
El rendimiento entre muebles se mide por separado en los benchmarks; las
trayectorias pueden variar entre equipos por diferencias numéricas.
