from sympy import symbols, sympify
import ast


def generate_c_expr(expr_terms):
    # TODO: should return a C expression string
    pass

def evaluate_expression_with_guard(expression):
    # Define symbolic variables
    # variables = symbols(','.join(variable_values.keys()))

    # Split the expression into terms
    terms = expression.split('+')
    pyterms = []

    # Evaluate each term separately
    result = 0
    print("Term list: {}".format(terms))
    terms = [term.strip() for term in terms]
    for term in terms:
        print(f"{term}")
        t = term.split('*', 1)
        if t[0][0] == '[':                
            guard_part, main_part = t[0], t[1]
            print(f'{guard_part}: {main_part}')

            # Parse the guard expression
            guard_expression = sympify(guard_part.strip().strip('[').strip(']'))
            print(f'guard: {guard_expression}')
            
            # Check the guard condition
            main_expression = sympify(main_part.strip())
            print(f'main simplified: {main_expression}')

            if main_expression == 0: continue
            
            pyterms.append((guard_expression, main_expression))
        else:
            # Parse the main expression
            main_part = term
            main_expression = sympify(main_part)
            pyterms.append((True, main_expression))
  
        
    return pyterms
    
    # except Exception as e:
    #     print(f"Error evaluating expression: {e}")
    #     return None

# Example expression with multiple terms and guard conditions
expression_with_guard = '[x > y] * x**2 * y * 0 + [x <= y] * y**2 * z'
# expression_with_guard = 'x**2 * y'



expr_terms = evaluate_expression_with_guard(expression_with_guard)
print(f"Expression terms: {expr_terms}")

cexpr = generate_c_expr(expr_terms)
# print(f"{cexpr}")
