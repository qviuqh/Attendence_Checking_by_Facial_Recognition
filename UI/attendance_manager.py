class AttendanceManager:
    def __init__(self, initial_data):
        # initial_data: dict[int, dict(name, id, present)]
        self.student_data = initial_data
        self.total_students = len(self.student_data)
        self.present_count = 0
        self.current_student = 1

    def mark_present(self, student_id):
        rec = self.student_data.get(student_id)
        if rec and not rec["present"]:
            rec["present"] = True
            self.present_count += 1
        # student = self.student_data.get(self.current_student)
        # if student and not student.get('present', False):
        #     student['present'] = True
        #     self.present_count += 1
        # self.advance()

    def advance(self):
        self.current_student += 1
        if self.current_student > len(self.student_data):
            self.current_student = 1

    def add_student(self, name, id_):
        new_id = len(self.student_data) + 1
        self.student_data[new_id] = {"name": name, "id": id_, "present": False}
        self.total_students += 1
        return new_id