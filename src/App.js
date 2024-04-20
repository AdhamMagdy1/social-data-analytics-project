import React, { useState } from 'react';
import './App.css';

function App() {
  const [sentence, setSentence] = useState('');
  const [sentenceResult, setSentenceResult] = useState('no result yet');
  const [course, setCourse] = useState('');
  const [courseResult, setCourseResult] = useState('no result yet');
  const [review, setReview] = useState('');
  const [university, setUniversity] = useState('');
  const [universityResult, setUniversityResult] = useState('no result yet');
  const [sentenceLoading, setSentenceLoading] = useState(false);
  const [courseLoading, setCourseLoading] = useState(false);
  const [universityLoading, setUniversityLoading] = useState(false);

  const courses = [
    'marketing in a digital world',
    'financial markets',
    'data analysis with python',
    'behavioral finance',
    'data science methodology',
    'getting started with google kubernetes engine',
    'introduction to marketing',
    'introduction to software product management',
    'programming fundamentals',
    'introduction to structured query language sql',
  ];

  const universities = ['MIT', 'Stanford', 'Harvard'];

  const handleSentenceChange = (event) => {
    setSentence(event.target.value);
  };

  const handleCourseChange = (event) => {
    setCourse(event.target.value);
  };

  const handleReviewChange = (event) => {
    setReview(event.target.value);
  };
  const handleUniversityChange = (event) => {
    setUniversity(event.target.value);
  };
  const handleSentenceProcess = async () => {
    setSentenceLoading(true);
    try {
      const response = await fetch('http:///localhost:5000/analyze_sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: sentence }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      setSentenceResult(data.vader_sentiment); 
    } catch (error) {
      console.error('Error processing sentence:', error);
      setSentenceResult('Error: Failed to process sentence.');
    } finally {
      setSentenceLoading(false);
    }
  };

  const handleCourseProcess = async () => {
    setCourseLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/recommendation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ course: course }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      setCourseResult(data.recommendation); 
    } catch (error) {
      console.error('Error processing course:', error);
      setCourseResult('Error: Failed to process course.');
    } finally {
      setCourseLoading(false);
    }
  };

  const handleUniversityProcess = async () => {
    setUniversityLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predictrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: review, institution: university }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      setUniversityResult(data.predicted_rate); 
    } catch (error) {
      console.error('Error processing review:', error);
      setUniversityResult('Error: Failed to process review.');
    } finally {
      setUniversityLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="box">
        <div className="divHolder">
          <label htmlFor="sentence">Enter your sentence</label>
          <input
            type="text"
            id="sentence"
            value={sentence}
            onChange={handleSentenceChange}
            placeholder="your sentence"
          />
        </div>
        <div className="btnHolder">
          <button onClick={handleSentenceProcess} disabled={sentenceLoading}>
            {sentenceLoading ? 'Loading...' : 'Process'}
          </button>
        </div>

        <div className="divHolder">
          <label className="res" htmlFor="sentence-result">
            Result
          </label>
          <div className="Result" id="sentence-result">
            {sentenceResult}
          </div>
        </div>
      </div>

      <div className="box">
        <div className="divHolder">
          <label htmlFor="course">Choose your course</label>
          <select id="course" value={course} onChange={handleCourseChange}>
            {courses.map((course) => (
              <option key={course} value={course}>
                {course}
              </option>
            ))}
          </select>
        </div>
        <div className="btnHolder">
          <button onClick={handleCourseProcess} disabled={courseLoading}>
            {courseLoading ? 'Loading...' : 'Process'}
          </button>
        </div>

        <div className="divHolder">
          <label className="res" htmlFor="course-result">
            Result
          </label>
          <div className="Result" id="course-result">
            {courseResult}
          </div>
        </div>
      </div>

      <div className="box">
        <div className="divHolder">
          <label htmlFor="review">Enter your review</label>
          <input
            id="review"
            value={review}
            onChange={handleReviewChange}
            placeholder="your review"
          />
        </div>
        <div className="divHolder">
          <label htmlFor="university">Choose your university</label>
          <select
            id="university"
            value={university}
            onChange={handleUniversityChange}
          >
            {universities.map((uni) => (
              <option key={uni} value={uni}>
                {uni}
              </option>
            ))}
          </select>
        </div>
        <div className="btnHolder">
          <button
            onClick={handleUniversityProcess}
            disabled={universityLoading}
          >
            {universityLoading ? 'Loading...' : 'Process'}
          </button>
        </div>
        <div className="divHolder">
          <label className="res" htmlFor="university-result">
            Result
          </label>
          <div className="Result" id="university-result">
            {universityResult}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
