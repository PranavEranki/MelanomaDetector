import coremltools
coreml_model = coremltools.converters.keras.convert('FinalModel.h5')
coremltools.utils.save_spec(coreml_model, 'testingmodel.mlmodel')
