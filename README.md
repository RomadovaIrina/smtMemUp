# SMT × MemUP: Stack без шума

* `02_smt_memup_stack.ipynb` — актуальный ноутбук: аудит, протокол, фазы эксперимента, предрегистрация (раздел 8.0).
* `smt_memup_stack.py` — весь код эксперимента; импорт ничего не обучает. Хэш файла входит в идентификатор запуска.
* `checks/test_smt_memup_stack.py` — технические проверки (CPU): `python checks/test_smt_memup_stack.py`.
* `checks/wait_and_run.sh` — запуск фаз на общем GPU: ждёт свободную память, resume при OOM, test не запускает.
* `01_smt_memup_preparation_archive.ipynb` — архив версии 01 с её сохранёнными результатами (не результаты кода 02).

## Окружение

```bash
uv venv --python 3.12 envs/smt-memup
uv pip install --python envs/smt-memup/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/smt-memup/bin/python numpy==2.1.0 pandas==2.3.3 matplotlib==3.10.8 einops==0.8.1 x-transformers==2.18.5 tqdm==4.67.1 ipykernel nbformat nbclient
```

## Авторские исходники (не входят в репозиторий)

```bash
git clone https://github.com/akarshkumar0101/smt.git upstream/smt && git -C upstream/smt checkout cf0d0f04be606ab9d97b1a449a6846450d656ad0
git clone https://github.com/griver/memup.git upstream/memup && git -C upstream/memup checkout d27657137f6e427e30fcd4ae885b20b071b2917d
```

## Фазы

```bash
python smt_memup_stack.py --phase audit|baseline|tune|final --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k
python smt_memup_stack.py --phase report --confirm-test --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k
```

Результаты и checkpoints пишутся в `runs/` (в репозиторий не входит).
