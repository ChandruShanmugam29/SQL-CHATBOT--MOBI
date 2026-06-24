import sqlite3
import random

conn = sqlite3.connect('company.db')
cursor = conn.cursor()

# 1. Departments Table
cursor.execute('DROP TABLE IF EXISTS departments')
cursor.execute('''CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT,
    head_of_dept TEXT
)''')

# 2. Employees Table (Linked with department)
cursor.execute('DROP TABLE IF EXISTS employees')
cursor.execute('''CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    dept_id INTEGER,
    salary INTEGER,
    designation TEXT,
    FOREIGN KEY(dept_id) REFERENCES departments(dept_id)
)''')

# 3. Projects Table
cursor.execute('DROP TABLE IF EXISTS projects')
cursor.execute('''CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT,
    emp_id INTEGER,
    FOREIGN KEY(emp_id) REFERENCES employees(id)
)''')

# Insert Departments
cursor.execute("INSERT INTO departments VALUES (1, 'IT', 'Nicholas Rush')")
cursor.execute("INSERT INTO departments VALUES (2, 'Sales', 'Thomas Lopez')")
cursor.execute("INSERT INTO departments VALUES (3, 'HR', 'Diana Moore')")
cursor.execute("INSERT INTO departments VALUES (4, 'Finance', 'Adam Lewis')")

employee_names = [
    "Jennifer Young",
    "Angela Fleming",
    "Keith Anthony",
    "Nicholas Rush",
    "Calvin Bruce",
    "Kimberly Gonzalez MD",
    "Amanda Carpenter",
    "Diana Hoover",
    "Dr. Andrew Arellano Jr.",
    "Thomas Lopez",
    "Diana Moore",
    "Amanda Hill",
    "Samuel Hill",
    "Brenda Love",
    "Adam Lewis",
    "Mrs. Mary Reid",
    "Jack Thompson",
]

# Medium-difficulty designations
designations = [
    "Senior Developer",
    "Data Analyst",
    "Project Manager",
    "Team Lead",
    "Business Analyst",
    "DevOps Engineer",
    "QA Engineer",
]

dept_ids = [1, 2, 3, 4]

random.seed(42)  # reproducible

project_names = [
    "Cloud Migration Initiative",
    "CRM Overhaul",
    "HR Digital Transformation",
    "Budget Automation Suite",
    "AI-Powered Analytics",
    "Customer Portal Redesign",
    "Cybersecurity Audit",
    "ERP Integration",
    "Mobile App Launch",
    "Data Warehouse Upgrade",
    "Compliance Management System",
    "Sales Pipeline Optimizer",
    "Employee Self-Service Portal",
    "Payroll System Upgrade",
    "Infrastructure Modernization",
    "Knowledge Management Platform",
    "Supplier Onboarding Tool",
]

for i, name in enumerate(employee_names, start=1):
    dept = random.choice(dept_ids)
    salary = random.randint(45000, 120000)
    designation = random.choice(designations)
    cursor.execute(
        "INSERT INTO employees VALUES (?,?,?,?,?)",
        (i, name, dept, salary, designation)
    )
    cursor.execute(
        "INSERT INTO projects VALUES (?,?,?)",
        (i, project_names[i - 1], i)
    )

conn.commit()
conn.close()
print("Database created successfully with fixed employee names and medium-difficulty data!")