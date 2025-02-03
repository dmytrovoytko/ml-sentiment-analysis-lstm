import requests
import sys
from settings import DATA_FILE, DEBUG, PORT, TARGET, TEST_FILE

DEBUG = True # True # False # override global settings

if __name__ == '__main__':
    # quick tests
    from preprocess import load_data
    # testing with {sample_size} records 
    sample_size = 10 
    print(f'\nTesting prediction on {sample_size} records from {TEST_FILE["FILE_NAME"]}...') # DATA_FILE

    df_full = load_data(TEST_FILE, verbose=False)
    df = df_full.head(sample_size)

    if len(sys.argv)>1 and sys.argv[1]=='--deployed':
        # Testing deployed service on huggingface
        api_url = f'https://dmytrovoytko-ml-sentiment-analysis.hf.space/'
    else:
        api_url = f'http://localhost:{PORT}/'

    print('Testing web service:', api_url)

    print('\nSingle value prediction 1 by 1:')
    url = api_url+'predict'
    for row in df.to_dict('records'):
        try:
            response = requests.post(url, json=row)
            print('\n data:', row)
            print(' response:', response.status_code)
            if response.status_code==200:
                print('   source:', row[TARGET], ' -> prediction:', response.json())
            else:
                print('   error:', response.text)
        except Exception as e:
            print('   error:', e)

    print('\nBatch (list) prediction:')
    url = api_url+'predict_list'
    text_list = df.to_dict('records')
    response = requests.post(url, json=text_list)
    print('\n data:', text_list)
    print(' response:', response.status_code)
    if response.status_code==200:
        print(' prediction:', response.json())
    else:
        print('   error:', response.text)

    sample_size = 100
    print(f'\nBatch prediction {sample_size} records with evaluation:')
    url = api_url+'evaluate_prediction'
    df = df_full.head(sample_size)
    text_list = df.to_dict('records')
    response = requests.post(url, json=text_list)
    print(' response:', response.status_code)
    if response.status_code==200:
        # import pprint
        # pprint.pprint(' prediction:', response.json())
        print("\n".join("{}\t{}".format(k, v) for k, v in response.json().items()))
    else:
        print('   error:', response.text)
