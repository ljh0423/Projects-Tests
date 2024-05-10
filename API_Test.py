from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

# Connect to SQLite database
conn = sqlite3.connect('students.db')
cursor = conn.cursor()

# Create a table for students if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT,
        grade TEXT
    )
''')

# Sample data
students = [
    {"id": 1, "name": "Alice", "grade": "A"},
    {"id": 2, "name": "Bob", "grade": "B"},
    {"id": 3, "name": "Charlie", "grade": "C"}
]

# Insert sample data into the database
for student in students:
    cursor.execute('''
        INSERT OR REPLACE INTO students (id, name, grade)
        VALUES (?, ?, ?)
    ''', (student["id"], student["name"], student["grade"]))

conn.commit()

# API endpoint to get all students
@app.route('/api/students', methods=['GET'])
def get_students():
    cursor.execute('SELECT * FROM students')
    students_data = cursor.fetchall()
    return jsonify(students_data)

# API endpoint to get a specific student by ID
@app.route('/api/students/<int:id>', methods=['GET'])
def get_student(id):
    cursor.execute('SELECT * FROM students WHERE id = ?', (id,))
    student = cursor.fetchone()
    if student:
        return jsonify(student)
    else:
        return jsonify({"error": "Student not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
