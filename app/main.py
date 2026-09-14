"""Local prediction API. Configure one completed Step 6 run before serving."""

from contextlib import asynccontextmanager
import csv
import io
import logging
import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.schemas import BatchRequest, BatchResult, PredictionRequest, PredictionResult, ProfileRequest, ProfileResult
from src.inference.registry import ModelRegistry, CONFIG_RELATIVE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/javascript', '.mjs')
logger = logging.getLogger('battery_voltage.api')


def create_app(project_root=PROJECT_ROOT, *, registry=None, allow_test_artifacts=False):
    root = Path(project_root)

    @asynccontextmanager
    async def lifespan(application):
        application.state.registry = None
        application.state.readiness_message = 'Configure a completed run with scripts/step7_configure_api.py.'
        if registry is not None:
            application.state.registry = registry
        elif (root / CONFIG_RELATIVE).exists():
            try:
                application.state.registry = ModelRegistry.from_config(root, allow_test_artifacts=allow_test_artifacts)
            except Exception:
                logger.exception('Model startup failed')
                application.state.readiness_message = 'Model loading failed. Inspect the server terminal for the error.'
        if application.state.registry is not None:
            application.state.readiness_message = 'All six configured model bundles are ready.'
        yield
        application.state.registry = None

    application = FastAPI(title='Battery Electrode Voltage API', version='0.8.0', lifespan=lifespan,
        description='Predict computed average insertion voltages using saved Step 6 models. No training or API key is needed for inference.')
    application.mount('/static', StaticFiles(directory=root / 'frontend' / 'assets', check_dir=False), name='static')

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': list(error['loc']), 'type': error['type'], 'msg': error['msg']} for error in exc.errors()]})

    def active(request):
        selected = request.app.state.registry
        if selected is None:
            raise HTTPException(status_code=503, detail=request.app.state.readiness_message)
        return selected

    def infer(request, items):
        selected = active(request)
        try:
            predictions = selected.predict(items)
        except Exception:
            logger.exception('Prediction failed')
            raise HTTPException(status_code=500, detail='Prediction failed. Inspect the server terminal; no substitute voltage was generated.') from None
        return {'run_id': selected.run_id, 'count': len(predictions), 'predictions': predictions}

    @application.get('/', include_in_schema=False)
    def home():
        return FileResponse(root / 'frontend' / 'index.html', headers={'Cache-Control': 'no-cache'})

    @application.get('/health', tags=['Status'])
    def health(request: Request):
        selected = request.app.state.registry
        return {'status': 'ok', 'stage': 'prediction_api', 'model_loaded': selected is not None,
                'run_id': selected.run_id if selected else None,
                'loaded_models': len(selected.bundles) if selected else 0,
                'message': request.app.state.readiness_message}

    @application.get('/ready', tags=['Status'])
    def ready(request: Request):
        selected = active(request)
        return {'ready': True, 'run_id': selected.run_id, 'loaded_models': len(selected.bundles)}

    @application.get('/models', tags=['Models'])
    def models(request: Request):
        return active(request).catalog()

    @application.post('/predict', response_model=PredictionResult, tags=['Prediction'])
    def predict(body: PredictionRequest, request: Request):
        return infer(request, [body])['predictions'][0]

    @application.post('/predict/batch', response_model=BatchResult, tags=['Prediction'])
    def batch(body: BatchRequest, request: Request):
        return infer(request, body.items)

    @application.post('/predict/batch.csv', response_class=Response, tags=['Prediction'],
                      responses={200: {'content': {'text/csv': {}}, 'description': 'Predictions as a downloadable CSV'}})
    def batch_csv(body: BatchRequest, request: Request):
        result = infer(request, body.items)
        text = io.StringIO(newline='')
        writer = csv.writer(text)
        writer.writerow(['row', 'working_ion', 'formula_charge', 'formula_discharge', 'predicted_voltage_V',
                         'model_family', 'training_population', 'run_id', 'extrapolates_working_ion', 'warnings'])
        for index, item in enumerate(result['predictions'], 1):
            writer.writerow([index, item['working_ion'], item['interval']['formula_charge'], item['interval']['formula_discharge'],
                item['predicted_voltage_V'], item['model']['family'], item['model']['training_population'],
                item['run_id'], item['extrapolates_working_ion'], ' | '.join(item['warnings'])])
        return Response(text.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition': 'attachment; filename="voltage_predictions.csv"'})

    @application.post('/predict/profile', response_model=ProfileResult, tags=['Prediction'])
    def profile(body: ProfileRequest, request: Request):
        return infer(request, body.intervals)

    return application


app = create_app()
