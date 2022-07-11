"""
Copyright [2009-present] EMBL-European Bioinformatics Institute
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Usage: python search.py [file] [database]
Examles:
python search.py file.fasta  # (search in the RNAcentral database)
python search.py file.fasta mirbase
"""

import json
import math
import requests  # pip install requests
import sys
import time
from Bio import SeqIO  # pip install biopython
from pathlib import Path

# sequence search server
server = "https://search.rnacentral.org/"


def get_sequence_search_result(description, sequence, job_id):
    """
    Function to check the status of the job and also to get the results of the sequence search
    :param description: description of the FASTA record
    :param sequence: sequence of the FASTA record
    :param job_id: id of the job
    :return: dict containing the results
    """
    get_status = requests.get(server + "api/job-status/" + job_id)
    job_status = json.loads(get_status.text)["status"]

    if job_status == "success" or job_status == "partial_success":
        # get results
        # there is a limit on the number of results that can be requested
        start = 0
        get_result = requests.get(server + "api/facets-search/" + job_id + "?start=" + str(start) + "&size=100")
        get_result = json.loads(get_result.text)
        results = [get_result["entries"]]

        # check the number of similar sequences and make new requests if necessary
        hit_count = get_result["hitCount"]
        iter_number = int(math.ceil(hit_count / 100.0))
        for num in range(iter_number - 1):
            start += 100
            new_request = requests.get(server + "api/facets-search/" + job_id + "?start=" + str(start) + "&size=100")
            new_request_result = json.loads(new_request.text)
            results.append(new_request_result["entries"])

        job_result = {
            "job_id": job_id,
            "hits": hit_count,
            "status": job_status,
            "description": description,
            "sequence": str(sequence),
            "results": [item for sublist in results for item in sublist]
        }

        return job_result

    elif job_status == "error" or job_status == "timeout":
        # return the metadata of the search
        # you can also send us the job_id so we can investigate further
        print("There was an error in the following record: {}".format(description))
        return {"job_id": job_id, "status": job_status, "description": description, "sequence": str(sequence)}

    elif job_status == "started" or job_status == "pending":
        # try again in 10 seconds
        time.sleep(10)
        return get_sequence_search_result(description, sequence, job_id)


def get_rfam_result(job_id):
    """
    Function to check the status of the Rfam job and also to get the results
    :param job_id: id of the job
    :return: rfam results
    """
    infernal_status = requests.get(server + "api/infernal-status/" + job_id)
    infernal_status = json.loads(infernal_status.text)["status"]
    results = []

    if infernal_status == "success":
        # get results
        infernal_results = requests.get(server + "api/infernal-result/" + job_id)
        infernal_results = json.loads(infernal_results.text)

        if infernal_results:
            for item in infernal_results:
                results.append({
                    "family": item["description"],
                    "accession": item["accession_rfam"],
                    "start": item["seq_from"],
                    "end": item["seq_to"],
                    "bit_score": item["score"],
                    "e_value": item["e_value"],
                    "strand": item["strand"],
                    "alignment": item["alignment"]
                })

        return results if results else "The query sequence did not match any Rfam families"

    elif infernal_status == "error" or infernal_status == "timeout":
        return "There was an error in the Rfam search"

    elif infernal_status == "started" or infernal_status == "pending":
        # try again in 10 seconds
        time.sleep(10)
        return get_rfam_result(job_id)


def main():
    filename = None
    database = None

    if len(sys.argv) == 1:
        print("You must specify the FASTA file")
        exit()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        database = []
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        database = [sys.argv[2]]
    else:
        print("Usage: python search.py file.fasta")
        exit()

    # create directory to store search results
    Path("results").mkdir(parents=True, exist_ok=True)

    with open(filename, mode='r') as handle:
        # use Biopython's parse function to process individual FASTA records
        for record in SeqIO.parse(handle, 'fasta'):
            # extract individual parts of the FASTA record
            description = record.description
            sequence = record.seq

            # submit a job
            data = {"databases": database, "query": str(sequence)}
            post_job = requests.post(server + "api/submit-job", json=data)

            # get job_id
            job_id = None
            if post_job.status_code == 201:
                job_id = json.loads(post_job.text)["job_id"]
            else:
                print("Failed to submit job. Record that failed:\n {} \n {}".format(description, sequence))

            if job_id:
                # get sequence search results
                sequence_search = get_sequence_search_result(description, sequence, job_id)

                # get Rfam results
                rfam = get_rfam_result(job_id)

                # joining results
                if sequence_search:
                    sequence_search["rfam_results"] = rfam

                # save results
                print("Saving results for {}".format(description))
                with open("results/" + description + '.json', 'w') as f:
                    json.dump(sequence_search, f)


if __name__ == "__main__":
    main()