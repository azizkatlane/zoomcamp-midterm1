import requests

url='http://127.0.0.1:4545/predict'

student_id='00000'
student = {
    'coursecategory' : 'Programming',
    'devicetype' : 'laptop',
    'numberofquizzestaken' : 3,
    'quizscores' : 100,
    'timespentoncourse' : 100,
    'numberofvideoswatched' : 100,
    'completionrate' : 100,
}


resp = requests.post(url,json=student).json()

print(resp)

if resp['course_completion']:
    print(f'{student_id} will complete the course')
else:
    print(f'{student_id} will not complete the course')