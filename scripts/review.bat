@echo off

:: move to project root
cd ..

:: format imports
isort . ./decodanda ./tests

:: run test suite
coverage run

:: export coverage to json / lcov for processing
coverage json
coverage lcov

:: export coverage to html for development in IDE
coverage html

:: report to console
coverage report

:: run linter (automatically goes to html for IDE due via configuration)
flake8
