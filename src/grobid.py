import requests



GROBID_URL = (
"http://localhost:8070/api/processFulltextDocument"
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