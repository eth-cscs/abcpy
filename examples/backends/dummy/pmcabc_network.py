# NOTE IF WE WANT TO ADD, WE MIGHT NOT WANT TO HAVE THESE SCORES THAT BIG!!!
# NOTE IF GRADES ARE +, THE SCHOLARSHIP SHOULD BE -? (or rather flipped)

from abcpy.continuousmodels import Normal, Uniform

# The prior information specifiying that depending on the school location, the score for class size and social background should be different
school_location_score = Uniform([[1,1],[1.5,1.7]])

# The school location affects both the class size and the social background
class_size_score = Normal([[school_location_score[0]], [0.1]])

background_score = Normal([[school_location_score[1]], [0.1]])

# Mean grade
grade = Normal([[4.5],[0.75]])

# The final model of the grade
model_1 = grade+class_size_score+background_score

#Probability that somebody gets scholarship
scholarship_score = Normal([[2],[0.5]])

# The final score, including the social background
model_2 = scholarship_score + background_score


