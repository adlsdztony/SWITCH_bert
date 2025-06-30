from social_worker_bert import SocialWorkerBERTPipeline

model_path = '/home/switch/switch_research/bert_classifier/SWITCH_bert/results_with_validation/social_worker_bert_model.pt'  # Update with your model path

pipeline = SocialWorkerBERTPipeline(config_path='./example_config.yaml')
pipeline.load_trained_model(model_path)

pipeline.process_data()
pipeline.evaluate_model()
pipeline.save_results('./results')