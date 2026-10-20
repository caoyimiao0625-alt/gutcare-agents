# GutCare 智能体服务 · 常用命令（零第三方依赖，不需要 pip install）
#
# 需要 make；Windows 上可直接用 run.ps1 或 README 里的命令。

PY ?= python
FIXTURE ?= $(abspath ../testdata_v4)
export GCA_FIXTURE_ROOT := $(FIXTURE)
export PYTHONIOENCODING := utf-8
PORT ?= 8100

.PHONY: help test eval samples serve demo demo-run smoke package check-docs docker clean

help:
	@echo "make test     运行 109 个回归用例"
	@echo "make eval     运行确定性评测（mock 模式）"
	@echo "make samples  重新生成联调样例 data/generated_samples/"
	@echo "make serve    启动服务与控制台（mock 模式，端口 $(PORT)）"
	@echo "make smoke    对已启动的服务做部署自检（PASS/FAIL）"
	@echo "make demo-run 跑一遍 6 步演示序列并落盘记录"
	@echo "make check-docs 校验文档里的仓库路径都存在"
	@echo "make package  打包提交件到 dist/"
	@echo "make docker   构建并启动容器（replay 模式）"

test:
	$(PY) -m unittest discover -s tests -v

eval:
	$(PY) eval/run_eval.py --mode mock

samples:
	$(PY) tools/make_samples.py

serve:
	GCA_MODE=mock GCA_PORT=$(PORT) $(PY) -m gutcare_agents serve

demo:
	GCA_MODE=replay GCA_REPLAY_FALLBACK=1 GCA_PORT=$(PORT) $(PY) -m gutcare_agents serve

demo-run:
	$(PY) tools/demo_run.py

smoke:
	$(PY) tools/smoke_test.py --base-url http://127.0.0.1:$(PORT)

package:
	$(PY) tools/package_submission.py

check-docs:
	$(PY) tools/check_docs.py --verbose

docker:
	docker compose up --build

clean:
	-rm -rf data/ dist/
	-find . -name "__pycache__" -type d -prune -exec rm -rf {} +
