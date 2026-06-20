import os, tempfile
from utils import save_model, load_model
print('utils imported, functions:', save_model.__name__, load_model.__name__)
# create simple objects
model = {'a':1}
vectorizer = {'v':2}
tmp_dir = tempfile.gettempdir()
model_path = os.path.join(tmp_dir, 'test_model_joblib_for_utils.joblib')
vec_path = os.path.join(tmp_dir, 'test_vec_joblib_for_utils.joblib')
# Save
save_model(model, vectorizer, model_path, vec_path)
print('Saved files:', model_path, vec_path)
# Load
loaded_model, loaded_vec = load_model(model_path, vec_path)
print('loaded_model:', loaded_model)
print('loaded_vec:', loaded_vec)
# cleanup
try:
    os.remove(model_path)
    os.remove(vec_path)
    print('Cleaned up temp files')
except Exception as e:
    print('Cleanup error:', e)
