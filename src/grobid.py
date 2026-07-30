import requests



GROBID_URL = (
"https://grobid.science-miner.com/api/processFulltextDocument"
)



def process_pdf(pdf_file):


    response = requests.post(

        GROBID_URL,

        files={
            "input":pdf_file
        },

        headers={
            "Accept":"application/xml"
        }

    )


    return response.text