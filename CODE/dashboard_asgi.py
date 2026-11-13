"""Deployment HTTP adapter. Run one Uvicorn worker behind the supplied Caddy gateway."""
import asyncio
from contextlib import asynccontextmanager
import json
from urllib.parse import parse_qs

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from dashboard_results import parse_filters, patients_csv, select_patients
from web_dashboard import Dashboard, ROOT, STATIC, dashboard_configuration, ensure_sumo

NO_CACHE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def json_response(data, status=200):
    return JSONResponse(data, status_code=status, headers=NO_CACHE)


def read_api(request):
    dashboard = request.app.state.dashboard
    path = request.url.path
    query = parse_qs(request.url.query)
    try:
        if path == "/api/config":
            return json_response(dashboard_configuration(dashboard))
        if path == "/api/state":
            return json_response(dashboard.snapshot())
        if path == "/api/network":
            if dashboard.network_error:
                return json_response({"error": dashboard.network_error}, 500)
            return json_response(dashboard.network if dashboard.network is not None else {"loading": True},
                                 200 if dashboard.network is not None else 202)
        if path == "/api/runs":
            return json_response({"runs": dashboard.saved_runs()})
        if path == "/api/patients":
            return json_response(dashboard.patient_snapshot(query))
        if path == "/api/patients.csv":
            records, _, _ = dashboard.patient_data(query)
            payload = patients_csv(select_patients(records, parse_filters(query)))
            return Response(payload, media_type="text/csv; charset=utf-8", headers={**NO_CACHE,
                            "Content-Disposition": 'attachment; filename="patient-transfers.csv"'})
        if path == "/api/export":
            payload = json.dumps(dashboard.export_snapshot(), ensure_ascii=False, allow_nan=False, indent=2)
            return Response(payload.encode("utf-8"), media_type="application/json; charset=utf-8", headers={**NO_CACHE,
                            "Content-Disposition": 'attachment; filename="ambulance-results.json"'})
        return json_response({"error": "API를 찾을 수 없습니다."}, 404)
    except ValueError as exc:
        return json_response({"error": str(exc)}, 400)


async def mutate_api(request):
    dashboard = request.app.state.dashboard
    if request.headers.get("X-Dashboard-Token") != dashboard.token:
        return json_response({"error": "화면을 새로고침한 뒤 다시 시도하세요."}, 403)
    if request.url.path not in ("/api/start", "/api/control"):
        return json_response({"error": "API를 찾을 수 없습니다."}, 404)
    chunks, size = [], 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > 8192:
                return json_response({"error": "요청 크기가 올바르지 않습니다."}, 400)
            chunks.append(chunk)
        body = json.loads(b"".join(chunks))
        if not isinstance(body, dict):
            raise ValueError("요청 형식이 올바르지 않습니다.")
        action = dashboard.start if request.url.path == "/api/start" else dashboard.control
        await run_in_threadpool(action, body)
        return json_response(dashboard.snapshot())
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        return json_response({"error": str(exc)}, 400)


def health(request):
    dashboard = request.app.state.dashboard
    ready = dashboard.network is not None and dashboard.network_error is None
    status = 200 if request.url.path == "/healthz" or ready else 503
    return json_response({"status": "ok" if status == 200 else "not_ready",
                          "network": "error" if dashboard.network_error else "ready" if ready else "loading"}, status)


def create_app(dashboard_factory=Dashboard):
    @asynccontextmanager
    async def lifespan(app):
        ensure_sumo()
        (ROOT / "results").mkdir(exist_ok=True)
        app.state.dashboard = dashboard_factory()
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.dashboard.close)

    return Starlette(debug=False, lifespan=lifespan, routes=[
        Route("/healthz", health), Route("/readyz", health),
        Route("/api/{path:path}", read_api, methods=["GET"]),
        Route("/api/{path:path}", mutate_api, methods=["POST"]),
        Mount("/", StaticFiles(directory=STATIC, html=True)),
    ])


app = create_app()
