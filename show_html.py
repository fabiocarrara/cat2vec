import argparse
import base64
import sys
import pandas as pd

from PIL import Image
from io import BytesIO
from urllib.parse import quote_plus as urlencode

def path2tag(path):
    im = Image.open(path)
    im.thumbnail((256, 256), Image.ANTIALIAS)
    im_buffer = BytesIO()
    im.save(im_buffer, format="JPEG")    
    data = base64.b64encode(im_buffer.getvalue()).decode('utf-8') #.replace('\n', '')
    return '<img src="data:image/png;base64,{0}">'.format(data)


def whatsthat(keywords):
    return '<a href="http://www.google.com/search?q={}" target="_blank">{}</a>'.format(urlencode(keywords), keywords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('predictions', type=str, help='CSV of predictions')
    parser.add_argument('-o', '--output', help='HTML output file')
    
    args = parser.parse_args()
    results = pd.read_csv(args.predictions)
    
    results['ILSVRC12Prediction'] = results['ILSVRC12Prediction'].apply(lambda x: ' '.join(x.split()[1:]))
    results = results[['URL', 'GoogleNewsScore', 'GoogleNewsPrediction', 'ILSVRC12Score', 'ILSVRC12Prediction', 'LabelCosineSimilarity']]
    
    results = results.sort_values('LabelCosineSimilarity', ascending=True)
    # results['ScoreDifference'] = (results['GoogleNewsScore'] - results['ILSVRC12Score'])**2
    # results = results.sort_values('ScoreDifference', ascending=False)
    # results = results[results.GoogleNewsScore > 0.75]  # get only good predictions
    # results = results.head(100)
    
    args.output = open(args.output, 'w') if args.output else sys.stdout
    
    with pd.option_context('display.max_colwidth', -1):
        formatters = dict(
            URL=path2tag,
            GoogleNewsPrediction=whatsthat,
            ILSVRC12Prediction=whatsthat
        )
        results.to_html(args.output, escape=False, max_cols=None, formatters=formatters)
        
    args.output.close()
    
