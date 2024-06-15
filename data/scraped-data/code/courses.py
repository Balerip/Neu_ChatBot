import json
from scraper import get_content

def get_courses_khoury(stream):
    match stream:
        case "CS":
            content = get_content("https://catalog.northeastern.edu/graduate/computer-information-science/computer-science/#coursestext")
            courses = content.find_all('div', class_='courseblock')
        case "CY":
            content = get_content("https://catalog.northeastern.edu/graduate/computer-information-science/cybersecurity/#coursestext")
            courses = content.find_all('div', class_='courseblock')
        case "DS":
            content = get_content("https://catalog.northeastern.edu/graduate/computer-information-science/data-science/#coursestext")
            courses = content.find_all('div', class_='courseblock')

    courses_data = []

    for block in courses:
        course = {}
        title = block.find("p", class_="courseblocktitle").text.strip()
        description = block.find("p", class_="cb_desc").text.strip()
        extras = block.find_all("p", class_="courseblockextra")
        corequisite = None
        prerequisite = None
        for extra in extras:
            text = extra.text.strip()
            if "Corequisite" in text:
                corequisite = text
            elif "Prerequisite" in text:
                prerequisite = text
        
        # Store data in a dictionary
        course['title'] = title
        course['description'] = description
        course['corequisite'] = corequisite
        course['prerequisite'] = prerequisite
        
        # Append the course dictionary to the courses list
        courses_data.append(course)

    courses_json = json.dumps(courses_data, indent=4)
    return courses_json

with open("../courses/courses_cs.json", "w") as cs_json_file:
    cs_json_file.write(get_courses_khoury("CS"))

with open("../courses/courses_cy.json", "w") as cy_json_file:
    cy_json_file.write(get_courses_khoury("CY"))

with open("../courses/courses_ds.json", "w") as ds_json_file:
    ds_json_file.write(get_courses_khoury("DS"))