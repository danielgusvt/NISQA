from run_predict import predict_mos

# Run prediction
results_df = predict_mos('weights/nisqa_mos_only.tar', 'MuestraVocal13.wav')

# Store MOS result in a variable
mos_pred = results_df['mos_pred'].iloc[0]
print(f"Predicted MOS: {mos_pred}")
