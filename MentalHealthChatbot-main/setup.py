from setuptools import setup, find_packages

setup(
    name='MentalHealthChatbot',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A chatbot for mental health support using NLP and Computer Vision',
    long_description=open(r'MentalHealthChatbot-main\README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/MentalHealthChatbot',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'spacy',
        'opencv-python',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
