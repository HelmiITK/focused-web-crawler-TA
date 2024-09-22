from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import urllib.parse

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    journals = []
    search_query = request.args.get('query')
    
    if search_query:
        encoded_query = urllib.parse.quote(search_query)
        url = f'https://scholar.google.com/scholar?hl=id&q={encoded_query}'
        response = requests.get(url)

        if response.status_code == 200:
            page_content = response.content
            soup = BeautifulSoup(page_content, 'html.parser')
            results = soup.find_all('div', class_='gs_r gs_or gs_scl')

            for result in results:
                title = result.find('h3', class_='gs_rt').text.strip()
                author = result.find('div', class_='gs_a').text.strip()
                abstract = result.find('div', class_='gs_rs').text.strip()
                
                # Menambahkan dictionary ke list journals
                journals.append({'title': title, 'author': author, 'abstract': abstract})

    return render_template('index.html', journals=journals)

if __name__ == '__main__':
    app.run(debug=True)
