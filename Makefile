exe_preprocessing:
	@echo "Executing Preprocessing Steps..."
	python src/preprocessing.py data/raw/ data/processed

gen_oversampler:
	@echo "Generates And Saving Synthetic Samples - OVERSAMPLER....."
	 python src/preprocessing_oversampler.py data/processed data/processed/sample_data

exe_train:
	@echo "Executing Train..."
	python ./src/train.py

run_application:
	@echo "Executing Preprocessing Steps..."
	streamlit run ./src/app.py


pipeline: exe_preprocessing gen_oversampler exe_train run_application