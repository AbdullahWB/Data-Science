-- MySQL Comprehensive Operations Example
-- Database Name: university_db

-- 1. Database Creation
-- This command creates a new database named 'university_db'.
-- If a database with the same name already exists, it will not throw an error.
CREATE DATABASE IF NOT EXISTS university_db;

-- Select the newly created database to perform operations within it.
USE university_db;

-- 2. Table Creation
-- We will create four tables: Students, Instructors, Courses, and Enrollments.

-- Table: Students
-- Stores information about students.
-- student_id: PRIMARY KEY, auto-incrementing for unique identification.
-- first_name, last_name: NOT NULL, VARCHAR for names.
-- date_of_birth: DATE type.
-- email: UNIQUE constraint to ensure no two students have the same email.
-- enrollment_date: DATE, with a DEFAULT value of the current date if not provided.
CREATE TABLE IF NOT EXISTS Students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE,
    email VARCHAR(100) UNIQUE,
    enrollment_date DATE DEFAULT (CURRENT_DATE)
);

-- Table: Instructors
-- Stores information about instructors.
-- instructor_id: PRIMARY KEY, auto-incrementing.
-- first_name, last_name: NOT NULL.
-- department: VARCHAR.
-- hire_date: DATE.
CREATE TABLE IF NOT EXISTS Instructors (
    instructor_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    department VARCHAR(100),
    hire_date DATE
);

-- Table: Courses
-- Stores information about courses.
-- course_id: PRIMARY KEY, auto-incrementing.
-- course_name: NOT NULL, UNIQUE to ensure no two courses have the same name.
-- credits: INT, with a CHECK constraint to ensure credits are between 1 and 6.
-- instructor_id: FOREIGN KEY, references instructor_id in the Instructors table.
-- ON DELETE SET NULL: If an instructor is deleted, their courses will have instructor_id set to NULL.
CREATE TABLE IF NOT EXISTS Courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL UNIQUE,
    credits INT NOT NULL,
    instructor_id INT,
    CONSTRAINT chk_credits CHECK (credits >= 1 AND credits <= 6), -- CHECK constraint
    FOREIGN KEY (instructor_id) REFERENCES Instructors(instructor_id) ON DELETE SET NULL
);

-- Table: Enrollments
-- Stores information about student enrollments in courses.
-- enrollment_id: PRIMARY KEY, auto-incrementing.
-- student_id: FOREIGN KEY, references student_id in the Students table.
-- course_id: FOREIGN KEY, references course_id in the Courses table.
-- enrollment_date: DATE.
-- grade: VARCHAR, can be NULL if not yet graded.
-- UNIQUE (student_id, course_id): Ensures a student can enroll in a course only once.
CREATE TABLE IF NOT EXISTS Enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    enrollment_date DATE NOT NULL,
    grade VARCHAR(2), -- e.g., 'A+', 'B', 'F'
    CONSTRAINT fk_student FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE, -- Cascade deletion
    CONSTRAINT fk_course FOREIGN KEY (course_id) REFERENCES Courses(course_id) ON DELETE CASCADE, -- Cascade deletion
    UNIQUE (student_id, course_id) -- Composite UNIQUE constraint
);

-- 3. Insert Data into Tables
-- Adding sample data to our tables.

-- Insert into Students
INSERT INTO Students (first_name, last_name, date_of_birth, email, enrollment_date) VALUES
('Alice', 'Smith', '2000-01-15', 'alice.smith@example.com', '2022-09-01'),
('Bob', 'Johnson', '1999-05-20', 'bob.j@example.com', '2021-09-01'),
('Charlie', 'Brown', '2001-11-10', 'charlie.b@example.com', '2022-01-15'),
('Diana', 'Prince', '2000-03-01', 'diana.p@example.com', '2023-09-01'),
('Eve', 'Adams', '1998-07-25', 'eve.a@example.com', '2021-01-01');

-- Insert into Instructors
INSERT INTO Instructors (first_name, last_name, department, hire_date) VALUES
('John', 'Doe', 'Computer Science', '2010-08-01'),
('Jane', 'Miller', 'Mathematics', '2015-01-10'),
('Peter', 'Jones', 'Physics', '2018-06-15');

-- Insert into Courses
INSERT INTO Courses (course_name, credits, instructor_id) VALUES
('Introduction to Programming', 3, (SELECT instructor_id FROM Instructors WHERE last_name = 'Doe')),
('Calculus I', 4, (SELECT instructor_id FROM Instructors WHERE last_name = 'Miller')),
('Quantum Mechanics', 3, (SELECT instructor_id FROM Instructors WHERE last_name = 'Jones')),
('Database Systems', 3, (SELECT instructor_id FROM Instructors WHERE last_name = 'Doe')),
('Linear Algebra', 3, (SELECT instructor_id FROM Instructors WHERE last_name = 'Miller'));

-- Insert into Enrollments
INSERT INTO Enrollments (student_id, course_id, enrollment_date, grade) VALUES
((SELECT student_id FROM Students WHERE last_name = 'Smith'), (SELECT course_id FROM Courses WHERE course_name = 'Introduction to Programming'), '2022-09-05', 'A'),
((SELECT student_id FROM Students WHERE last_name = 'Johnson'), (SELECT course_id FROM Courses WHERE course_name = 'Calculus I'), '2021-09-10', 'B+'),
((SELECT student_id FROM Students WHERE last_name = 'Brown'), (SELECT course_id FROM Courses WHERE course_name = 'Introduction to Programming'), '2022-01-20', 'C'),
((SELECT student_id FROM Students WHERE last_name = 'Prince'), (SELECT course_id FROM Courses WHERE course_name = 'Database Systems'), '2023-09-01', NULL),
((SELECT student_id FROM Students WHERE last_name = 'Smith'), (SELECT course_id FROM Courses WHERE course_name = 'Database Systems'), '2023-09-02', 'A-');

-- 4. Basic Data Manipulation Language (DML) Operations

-- SELECT: Retrieve all data from a table
SELECT * FROM Students;
SELECT * FROM Instructors;
SELECT * FROM Courses;
SELECT * FROM Enrollments;

-- SELECT with WHERE clause: Retrieve specific data
SELECT * FROM Students WHERE enrollment_date >= '2022-01-01';
SELECT first_name, last_name FROM Instructors WHERE department = 'Computer Science';

-- UPDATE: Modify existing data
-- Update Alice Smith's email address
UPDATE Students
SET email = 'alice.s@newdomain.com'
WHERE first_name = 'Alice' AND last_name = 'Smith';

SELECT * FROM Students WHERE first_name = 'Alice'; -- Verify update

-- Update the grade for Charlie Brown in Introduction to Programming
UPDATE Enrollments
SET grade = 'B-'
WHERE student_id = (SELECT student_id FROM Students WHERE first_name = 'Charlie' AND last_name = 'Brown')
AND course_id = (SELECT course_id FROM Courses WHERE course_name = 'Introduction to Programming');

SELECT * FROM Enrollments WHERE student_id = (SELECT student_id FROM Students WHERE first_name = 'Charlie' AND last_name = 'Brown'); -- Verify update

-- DELETE: Remove data from a table
-- Delete a student (e.g., Eve Adams)
-- Note: Due to ON DELETE CASCADE on Enrollments, her enrollments will also be deleted.
DELETE FROM Students
WHERE first_name = 'Eve' AND last_name = 'Adams';

SELECT * FROM Students; -- Verify deletion
SELECT * FROM Enrollments; -- Verify cascade deletion (Eve's enrollments should be gone)

-- 5. Showing Table Structure
-- DESCRIBE or SHOW COLUMNS FROM: View the schema (columns, data types, constraints) of a table.
DESCRIBE Students;
SHOW COLUMNS FROM Courses;

-- 6. Transactions
-- A transaction is a sequence of operations performed as a single logical unit of work.
-- Either all operations in the transaction are completed (committed), or none are (rolled back).

-- Scenario: Enroll a new student and assign them to a course.
START TRANSACTION;

-- Step 1: Insert a new student
INSERT INTO Students (first_name, last_name, date_of_birth, email, enrollment_date) VALUES
('Frank', 'White', '2002-09-01', 'frank.w@example.com', CURRENT_DATE());

-- Get the ID of the newly inserted student
SET @new_student_id = LAST_INSERT_ID();

-- Step 2: Enroll the new student in 'Calculus I'
-- We'll assume Calculus I exists, if not, this would fail.
SET @calculus_course_id = (SELECT course_id FROM Courses WHERE course_name = 'Calculus I');

INSERT INTO Enrollments (student_id, course_id, enrollment_date, grade) VALUES
(@new_student_id, @calculus_course_id, CURRENT_DATE(), NULL);

-- Check if both operations were successful.
-- If everything looks good, commit the transaction.
COMMIT;
-- If something went wrong or you want to undo, use ROLLBACK;
-- ROLLBACK;

SELECT * FROM Students WHERE first_name = 'Frank';
SELECT * FROM Enrollments WHERE student_id = @new_student_id;

-- Example of a transaction with ROLLBACK
START TRANSACTION;

-- Attempt to insert a student with an existing email (will cause an error due to UNIQUE constraint)
INSERT INTO Students (first_name, last_name, date_of_birth, email, enrollment_date) VALUES
('Grace', 'Hopper', '1906-12-09', 'alice.s@newdomain.com', CURRENT_DATE()); -- This email already exists!

-- If the above insert fails, the transaction will be rolled back.
-- In a real application, you'd check for errors and then decide to COMMIT or ROLLBACK.
-- For demonstration, we'll just force a rollback.
ROLLBACK;

SELECT * FROM Students WHERE first_name = 'Grace'; -- Grace Hopper should NOT be in the table.

-- 7. DELIMITER Command
-- The DELIMITER command is used to change the standard SQL statement delimiter (which is typically a semicolon ';').
-- This is essential when defining stored procedures, functions, or triggers, as these blocks of code
-- often contain multiple SQL statements, each terminated by a semicolon.
-- If the delimiter wasn't changed, the MySQL client would interpret the internal semicolons as the end of the
-- procedure/function/trigger definition, leading to syntax errors.

-- Example: Changing the delimiter to '//' before defining a trigger, then changing it back to ';'
-- (This is already used in the Triggers section, but explicitly demonstrating it here for clarity)

-- Change the delimiter to '//'
DELIMITER //

-- Create a simple stored procedure (for demonstration of DELIMITER usage)
-- This procedure will simply select all students.
CREATE PROCEDURE GetAllStudents()
BEGIN
    SELECT * FROM Students;
END // -- Note the new delimiter here

-- Change the delimiter back to ';'
DELIMITER ;

-- Now you can call the stored procedure
CALL GetAllStudents();

-- 8. Triggers
-- A trigger is a special type of stored procedure that automatically runs when a specific event occurs
-- in the database (INSERT, UPDATE, DELETE) on a specific table.

-- Example Trigger: Update a student's 'status' (simulated) or log changes.
-- Let's create a simple log table for enrollment changes.
CREATE TABLE IF NOT EXISTS Enrollment_Log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT,
    course_id INT,
    old_grade VARCHAR(2),
    new_grade VARCHAR(2),
    change_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    operation_type VARCHAR(10)
);

-- Trigger: BEFORE UPDATE on Enrollments
-- This trigger will log the old and new grade whenever an enrollment record is updated.
DELIMITER //
CREATE TRIGGER before_enrollment_update
BEFORE UPDATE ON Enrollments
FOR EACH ROW
BEGIN
    IF OLD.grade IS DISTINCT FROM NEW.grade THEN -- Only log if grade actually changes
        INSERT INTO Enrollment_Log (student_id, course_id, old_grade, new_grade, operation_type)
        VALUES (OLD.student_id, OLD.course_id, OLD.grade, NEW.grade, 'UPDATE');
    END IF;
END;
//
DELIMITER ;

-- Trigger: AFTER INSERT on Enrollments
-- This trigger could, for example, increment a counter for courses taken by a student.
-- For simplicity, let's just log the new enrollment.
DELIMITER //
CREATE TRIGGER after_enrollment_insert
AFTER INSERT ON Enrollments
FOR EACH ROW
BEGIN
    INSERT INTO Enrollment_Log (student_id, course_id, new_grade, operation_type)
    VALUES (NEW.student_id, NEW.course_id, NEW.grade, 'INSERT');
END;
//
DELIMITER ;

-- Test the triggers
-- First, update an existing enrollment's grade
UPDATE Enrollments
SET grade = 'A'
WHERE student_id = (SELECT student_id FROM Students WHERE first_name = 'Alice')
AND course_id = (SELECT course_id FROM Courses WHERE course_name = 'Introduction to Programming');

-- Now, insert a new enrollment to test the AFTER INSERT trigger
INSERT INTO Enrollments (student_id, course_id, enrollment_date, grade) VALUES
((SELECT student_id FROM Students WHERE first_name = 'Bob'), (SELECT course_id FROM Courses WHERE course_name = 'Database Systems'), CURRENT_DATE(), NULL);

SELECT * FROM Enrollment_Log; -- Check the log table

-- 9. Getting Current Date and Time
SELECT CURDATE() AS CurrentDate; -- Current date
SELECT CURRENT_DATE() AS CurrentDateAlias; -- Another alias for current date
SELECT CURTIME() AS CurrentTime; -- Current time
SELECT NOW() AS CurrentDateTime; -- Current date and time
SELECT SYSDATE() AS SystemDateTime; -- System date and time (can be different from NOW() in some contexts)

-- Date and Time Formatting
SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s') AS FormattedDateTime;
SELECT DATE_FORMAT(CURRENT_DATE(), '%M %d, %Y') AS FormattedDate;

-- Date Arithmetic
SELECT DATEDIFF(CURRENT_DATE(), '2023-01-01') AS DaysSinceNewYear;
SELECT ADDDATE(CURRENT_DATE(), INTERVAL 7 DAY) AS DateNextWeek;
SELECT SUBDATE(CURRENT_DATE(), INTERVAL 1 MONTH) AS DateLastMonth;

-- Extracting parts of a date
SELECT YEAR(NOW()) AS CurrentYear;
SELECT MONTH(NOW()) AS CurrentMonth;
SELECT DAY(NOW()) AS CurrentDay;

-- 10. Constraints (already demonstrated in CREATE TABLE, but let's re-emphasize)
-- PRIMARY KEY: student_id in Students, instructor_id in Instructors, etc. (Unique and NOT NULL)
-- FOREIGN KEY: instructor_id in Courses, student_id and course_id in Enrollments (Links tables)
-- UNIQUE: email in Students, course_name in Courses (Ensures uniqueness of values in a column)
-- NOT NULL: first_name, last_name in Students (Ensures a column cannot have NULL values)
-- DEFAULT: enrollment_date in Students (Provides a default value if not specified)
-- CHECK: credits in Courses (Ensures values meet a specific condition)

-- Attempt to violate a UNIQUE constraint (will fail)
-- INSERT INTO Students (first_name, last_name, date_of_birth, email, enrollment_date) VALUES
-- ('Test', 'User', '2000-01-01', 'alice.s@newdomain.com', CURRENT_DATE());
-- Error: Duplicate entry 'alice.s@newdomain.com' for key 'students.email'

-- Attempt to violate a NOT NULL constraint (will fail)
-- INSERT INTO Students (first_name, last_name, email) VALUES
-- (NULL, 'Test', 'test@example.com');
-- Error: Column 'first_name' cannot be null

-- Attempt to violate a CHECK constraint (will fail)
-- INSERT INTO Courses (course_name, credits, instructor_id) VALUES
-- ('Invalid Course', 0, (SELECT instructor_id FROM Instructors WHERE last_name = 'Doe'));
-- Error: Check constraint 'chk_credits' is violated.

-- 11. Joins
-- Combining rows from two or more tables based on a related column between them.

-- INNER JOIN: Returns rows when there is a match in both tables.
-- Get students and the courses they are enrolled in.
SELECT
    S.first_name,
    S.last_name,
    C.course_name,
    E.grade
FROM
    Students S
INNER JOIN
    Enrollments E ON S.student_id = E.student_id
INNER JOIN
    Courses C ON E.course_id = C.course_id;

-- LEFT JOIN (or LEFT OUTER JOIN): Returns all rows from the left table, and the matching rows from the right table.
-- If there is no match, NULL is returned for the right table's columns.
-- Get all students and their enrollments (if any). Students with no enrollments will still appear.
SELECT
    S.first_name,
    S.last_name,
    C.course_name,
    E.grade
FROM
    Students S
LEFT JOIN
    Enrollments E ON S.student_id = E.student_id
LEFT JOIN
    Courses C ON E.course_id = C.course_id;

-- RIGHT JOIN (or RIGHT OUTER JOIN): Returns all rows from the right table, and the matching rows from the left table.
-- If there is no match, NULL is returned for the left table's columns.
-- Get all courses and the instructors teaching them (if any). Courses without an instructor will still appear.
SELECT
    C.course_name,
    C.credits,
    I.first_name AS InstructorFirstName,
    I.last_name AS InstructorLastName
FROM
    Courses C
RIGHT JOIN
    Instructors I ON C.instructor_id = I.instructor_id;

-- FULL OUTER JOIN (MySQL does not directly support FULL OUTER JOIN. It can be simulated using UNION of LEFT and RIGHT JOINs)
-- Get all students and all courses, showing enrollments where they exist.
SELECT
    S.first_name,
    S.last_name,
    C.course_name,
    E.grade
FROM
    Students S
LEFT JOIN
    Enrollments E ON S.student_id = E.student_id
LEFT JOIN
    Courses C ON E.course_id = C.course_id
UNION
SELECT
    S.first_name,
    S.last_name,
    C.course_name,
    E.grade
FROM
    Students S
RIGHT JOIN
    Enrollments E ON S.student_id = E.student_id
RIGHT JOIN
    Courses C ON E.course_id = C.course_id;


-- 12. UNION
-- Combines the result sets of two or more SELECT statements.
-- Each SELECT statement within UNION must have the same number of columns,
-- the columns must have similar data types, and they must be in the same order.

-- Get the first names and last names of all students and instructors.
SELECT first_name, last_name FROM Students
UNION
SELECT first_name, last_name FROM Instructors;

-- UNION ALL: Includes duplicate rows (UNION removes duplicates by default)
SELECT first_name, last_name FROM Students
UNION ALL
SELECT first_name, last_name FROM Instructors;

-- 13. Functions
-- MySQL provides many built-in functions for various purposes.

-- Aggregate Functions: Perform calculations on a set of rows and return a single summary row.
SELECT COUNT(*) AS TotalStudents FROM Students;
SELECT AVG(credits) AS AverageCourseCredits FROM Courses;
SELECT SUM(credits) AS TotalCreditsOffered FROM Courses;
SELECT MIN(date_of_birth) AS OldestStudentDOB FROM Students;
SELECT MAX(enrollment_date) AS LatestEnrollmentDate FROM Students;

-- String Functions: Work with string data.
SELECT CONCAT(first_name, ' ', last_name) AS FullName FROM Students;
SELECT UPPER(course_name) AS CourseNameUpperCase FROM Courses;
SELECT LOWER(email) AS EmailLowerCase FROM Students;
SELECT LENGTH(department) AS DepartmentNameLength FROM Instructors;
SELECT SUBSTRING(course_name, 1, 5) AS CourseNamePrefix FROM Courses; -- Get first 5 characters

-- Numeric Functions: Work with numeric data.
SELECT ROUND(AVG(credits), 2) AS RoundedAverageCredits FROM Courses;
SELECT ABS(-100) AS AbsoluteValue;

-- Date Functions (already covered some, but more examples):
SELECT DAYNAME(CURRENT_DATE()) AS DayOfWeek;
SELECT MONTHNAME(CURRENT_DATE()) AS MonthOfYear;
SELECT YEAR('2023-11-25') AS YearOnly;

-- Conditional Functions:
SELECT
    first_name,
    last_name,
    IF(grade IS NULL, 'Not Graded', grade) AS FinalGrade
FROM
    Students S
JOIN
    Enrollments E ON S.student_id = E.student_id
WHERE E.grade IS NULL;

-- 14. Views
-- A view is a virtual table based on the result-set of an SQL statement.
-- A view contains rows and columns, just like a real table. The fields in a view are
-- fields from one or more real tables in the database.

-- Create a view showing student enrollment details
CREATE VIEW StudentEnrollmentDetails AS
SELECT
    S.student_id,
    S.first_name AS StudentFirstName,
    S.last_name AS StudentLastName,
    C.course_name,
    C.credits,
    I.first_name AS InstructorFirstName,
    I.last_name AS InstructorLastName,
    E.enrollment_date,
    E.grade
FROM
    Students S
JOIN
    Enrollments E ON S.student_id = E.student_id
JOIN
    Courses C ON E.course_id = C.course_id
JOIN
    Instructors I ON C.instructor_id = I.instructor_id;

-- Query the view just like a table
SELECT * FROM StudentEnrollmentDetails;

-- Filter the view
SELECT * FROM StudentEnrollmentDetails WHERE StudentLastName = 'Smith';

-- 15. Indexes
-- Indexes are used to retrieve data from the database more quickly.
-- The users cannot see the indexes, they are just used to speed up searches/queries.
-- PRIMARY KEY and UNIQUE constraints automatically create indexes.

-- Create a non-unique index on the last_name column of the Students table
CREATE INDEX idx_student_lastname ON Students (last_name);

-- Create a composite index on course_name and credits for faster course lookups
CREATE INDEX idx_course_name_credits ON Courses (course_name, credits);

-- You can view existing indexes (though not always necessary for general use)
SHOW INDEX FROM Students;
SHOW INDEX FROM Courses;

-- 16. Subqueries
-- A subquery (inner query or nested query) is a query embedded inside another SQL query.
-- It can be used in the SELECT, FROM, WHERE, and HAVING clauses.

-- Subquery in WHERE clause: Find students enrolled in 'Database Systems'
SELECT first_name, last_name
FROM Students
WHERE student_id IN (
    SELECT student_id
    FROM Enrollments
    WHERE course_id = (SELECT course_id FROM Courses WHERE course_name = 'Database Systems')
);

-- Subquery in SELECT clause: Get the number of courses each instructor teaches
SELECT
    I.first_name,
    I.last_name,
    (SELECT COUNT(*) FROM Courses C WHERE C.instructor_id = I.instructor_id) AS NumberOfCoursesTaught
FROM
    Instructors I;

-- Subquery in FROM clause (Derived Table): Find students who enrolled after 2022
SELECT
    derived_students.first_name,
    derived_students.last_name
FROM
    (SELECT student_id, first_name, last_name FROM Students WHERE enrollment_date > '2022-01-01') AS derived_students;

-- Using EXISTS with a subquery: Find instructors who teach at least one course
SELECT first_name, last_name
FROM Instructors I
WHERE EXISTS (
    SELECT 1
    FROM Courses C
    WHERE C.instructor_id = I.instructor_id
);

-- 17. GROUP BY and HAVING
-- GROUP BY: Groups rows that have the same values in specified columns into a summary row.
-- HAVING: Filters groups based on a specified condition (similar to WHERE, but for groups).

-- Count the number of students per enrollment year
SELECT
    YEAR(enrollment_date) AS EnrollmentYear,
    COUNT(student_id) AS NumberOfStudents
FROM
    Students
GROUP BY
    EnrollmentYear
ORDER BY
    EnrollmentYear;

-- Find departments with more than 1 instructor
SELECT
    department,
    COUNT(instructor_id) AS NumberOfInstructors
FROM
    Instructors
GROUP BY
    department
HAVING
    COUNT(instructor_id) > 1;

-- Calculate the average grade for each course (assuming grades are numeric for AVG, or using COUNT for non-numeric)
-- For demonstration, let's count students per course
SELECT
    C.course_name,
    COUNT(E.student_id) AS EnrolledStudents
FROM
    Courses C
JOIN
    Enrollments E ON C.course_id = E.course_id
GROUP BY
    C.course_name
HAVING
    COUNT(E.student_id) > 1; -- Only show courses with more than 1 student

-- 18. LIKE Operator
-- Used in a WHERE clause to search for a specified pattern in a column.
-- %: Represents zero, one, or multiple characters.
-- _: Represents a single character.

-- Find students whose first name starts with 'A'
SELECT first_name, last_name FROM Students WHERE first_name LIKE 'A%';

-- Find courses containing 'Program' anywhere in their name
SELECT course_name FROM Courses WHERE course_name LIKE '%Program%';

-- Find instructors whose last name has 'o' as the second letter
SELECT first_name, last_name FROM Instructors WHERE last_name LIKE '_o%';

-- Find emails ending with '.com'
SELECT email FROM Students WHERE email LIKE '%.com';

-- Clean up (Optional): Drop the database and log table
-- Use these commands only if you want to remove the created database and tables.
-- DROP DATABASE university_db;
-- DROP TABLE Enrollment_Log;
