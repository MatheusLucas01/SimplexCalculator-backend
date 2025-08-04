# app.py - Backend Python Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)  # Permitir requisições do React (localhost:3000)

class SimplexSolver:
    def __init__(self):
        self.tolerance = 1e-6
    
    def solve_problem(self, objective_coeffs, constraints, constraint_bounds, problem_type='maximize'):
        """
        Resolve problema de programação linear e gera gráfico
        
        Args:
            objective_coeffs: [c1, c2] - coeficientes da função objetivo
            constraints: [[a11, a12], [a21, a22], ...] - matriz de restrições
            constraint_bounds: [b1, b2, ...] - lados direitos
            problem_type: 'maximize' ou 'minimize'
        
        Returns:
            dict com solução e gráfico em base64
        """
        
        try:
            # Encontrar todos os vértices candidatos
            vertices = self._find_all_vertices(constraints, constraint_bounds)
            
            if not vertices:
                return {
                    'success': False,
                    'error': 'Nenhuma solução viável encontrada',
                    'vertices': [],
                    'optimal_point': None,
                    'optimal_value': None,
                    'graph': None
                }
            
            # Avaliar função objetivo em cada vértice
            evaluated_vertices = []
            for vertex in vertices:
                x1, x2 = vertex
                obj_value = objective_coeffs[0] * x1 + objective_coeffs[1] * x2
                evaluated_vertices.append({
                    'point': vertex,
                    'x1': x1,
                    'x2': x2,
                    'objective_value': obj_value,
                    'is_feasible': self._is_feasible_point(x1, x2, constraints, constraint_bounds)
                })
            
            # Filtrar apenas vértices viáveis
            feasible_vertices = [v for v in evaluated_vertices if v['is_feasible']]
            
            if not feasible_vertices:
                return {
                    'success': False,
                    'error': 'Problema inviável - nenhum vértice satisfaz todas as restrições',
                    'vertices': evaluated_vertices,
                    'optimal_point': None,
                    'optimal_value': None,
                    'graph': None
                }
            
            # Encontrar solução ótima
            if problem_type == 'maximize':
                optimal_vertex = max(feasible_vertices, key=lambda v: v['objective_value'])
            else:
                optimal_vertex = min(feasible_vertices, key=lambda v: v['objective_value'])
            
            # Gerar gráfico profissional
            graph_base64 = self._generate_professional_graph(
                objective_coeffs, constraints, constraint_bounds, 
                feasible_vertices, optimal_vertex, problem_type
            )
            
            return {
                'success': True,
                'optimal_point': {
                    'x1': optimal_vertex['x1'],
                    'x2': optimal_vertex['x2'],
                    'objective_value': optimal_vertex['objective_value']
                },
                'vertices': feasible_vertices,
                'all_vertices': evaluated_vertices,  # Inclui inviáveis para debug
                'graph': graph_base64,
                'problem_type': problem_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Erro interno: {str(e)}',
                'vertices': [],
                'optimal_point': None,
                'optimal_value': None,
                'graph': None
            }
    
    def _find_all_vertices(self, constraints, bounds):
        """Encontra todos os vértices candidatos (interseções)"""
        vertices = []
        
        # Origem (0,0)
        vertices.append((0.0, 0.0))
        
        # Interseções com eixos
        for i, (constraint, bound) in enumerate(zip(constraints, bounds)):
            a1, a2 = constraint
            
            # Interseção com eixo x1 (x2 = 0)
            if abs(a1) > self.tolerance:
                x1_intercept = bound / a1
                if x1_intercept >= -self.tolerance:  # Não-negativo
                    vertices.append((max(0.0, x1_intercept), 0.0))
            
            # Interseção com eixo x2 (x1 = 0)
            if abs(a2) > self.tolerance:
                x2_intercept = bound / a2
                if x2_intercept >= -self.tolerance:  # Não-negativo
                    vertices.append((0.0, max(0.0, x2_intercept)))
        
        # Interseções entre pares de restrições
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                intersection = self._line_intersection(
                    constraints[i], bounds[i], 
                    constraints[j], bounds[j]
                )
                
                if intersection:
                    x1, x2 = intersection
                    # Adicionar se estiver no quadrante não-negativo
                    if x1 >= -self.tolerance and x2 >= -self.tolerance:
                        vertices.append((max(0.0, x1), max(0.0, x2)))
        
        # Remover duplicatas
        unique_vertices = []
        for vertex in vertices:
            is_duplicate = False
            for existing in unique_vertices:
                if (abs(vertex[0] - existing[0]) < self.tolerance and 
                    abs(vertex[1] - existing[1]) < self.tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_vertices.append(vertex)
        
        return unique_vertices
    
    def _line_intersection(self, constraint1, bound1, constraint2, bound2):
        """Calcula interseção entre duas linhas"""
        a1, b1 = constraint1
        a2, b2 = constraint2
        
        # Resolver sistema: a1*x + b1*y = bound1, a2*x + b2*y = bound2
        det = a1 * b2 - a2 * b1
        
        if abs(det) < self.tolerance:
            return None  # Linhas paralelas
        
        x = (bound1 * b2 - bound2 * b1) / det
        y = (a1 * bound2 - a2 * bound1) / det
        
        return (x, y)
    
    def _is_feasible_point(self, x1, x2, constraints, bounds):
        """Verifica se um ponto satisfaz todas as restrições"""
        # Verificar não-negatividade
        if x1 < -self.tolerance or x2 < -self.tolerance:
            return False
        
        # Verificar cada restrição
        for constraint, bound in zip(constraints, bounds):
            a1, a2 = constraint
            value = a1 * x1 + a2 * x2
            if value > bound + self.tolerance:  # Violação da restrição <=
                return False
        
        return True
    
    def _generate_professional_graph(self, objective_coeffs, constraints, constraint_bounds, 
                                   feasible_vertices, optimal_vertex, problem_type):
        """Gera gráfico profissional estilo livro didático"""
        
        # Configurar figura
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calcular limites do gráfico
        all_x = [v['x1'] for v in feasible_vertices] + [0]
        all_y = [v['x2'] for v in feasible_vertices] + [0]
        
        max_x = max(all_x) * 1.3 if all_x else 100
        max_y = max(all_y) * 1.3 if all_y else 100
        
        # Ajustar para valores mínimos visualizáveis
        max_x = max(max_x, 10)
        max_y = max(max_y, 10)
        
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        
        # Grid profissional
        ax.grid(True, alpha=0.3, linewidth=1.2, color='gray')
        ax.set_axisbelow(True)
        
        # Eixos e labels
        ax.set_xlabel('X₁', fontsize=18, fontweight='bold', color='black')
        ax.set_ylabel('X₂', fontsize=18, fontweight='bold', color='black')
        ax.set_title('Resultado - Método Gráfico', fontsize=20, fontweight='bold', 
                    color='black', pad=20)
        
        # Cores profissionais
        constraint_colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
        
        # Plotar linhas das restrições
        x_line = np.linspace(0, max_x * 1.2, 1000)
        
        for i, (constraint, bound) in enumerate(zip(constraints, constraint_bounds)):
            a1, a2 = constraint
            color = constraint_colors[i % len(constraint_colors)]
            
            if abs(a2) > self.tolerance:
                # Linha não vertical: a1*x1 + a2*x2 = bound
                y_line = (bound - a1 * x_line) / a2
                
                # Plotar apenas onde y >= 0
                valid_mask = (y_line >= 0) & (y_line <= max_y * 1.1) & (x_line >= 0)
                if np.any(valid_mask):
                    ax.plot(x_line[valid_mask], y_line[valid_mask], 
                           color=color, linewidth=4, 
                           label=f'Restricción {i+1}: {a1}X₁ + {a2}X₂ ≤ {bound}')
            
            elif abs(a1) > self.tolerance:
                # Linha vertical: x1 = bound/a1
                x_vert = bound / a1
                if 0 <= x_vert <= max_x:
                    ax.axvline(x=x_vert, color=color, linewidth=4,
                              label=f'Restricción {i+1}: {a1}X₁ ≤ {bound}')
        
        # Região viável (polígono)
        if len(feasible_vertices) >= 3:
            # Ordenar vértices para formar polígono convexo
            points = [(v['x1'], v['x2']) for v in feasible_vertices]
            
            # Encontrar centro geométrico
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            
            # Ordenar por ângulo polar
            points.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            
            # Criar polígono
            feasible_region = Polygon(points, alpha=0.25, facecolor='lightblue', 
                                    edgecolor='#1f77b4', linewidth=3)
            ax.add_patch(feasible_region)
        
        # Plotar vértices
        vertex_labels = ['D', 'C', 'B', 'A']  # Ordem típica: origem, eixos, intersecção
        
        # Ordenar vértices para rotulação consistente
        sorted_vertices = sorted(feasible_vertices, key=lambda v: (v['x1'], v['x2']))
        
        for i, vertex in enumerate(sorted_vertices):
            x1, x2 = vertex['x1'], vertex['x2']
            is_optimal = (vertex == optimal_vertex)
            
            # Estilo do ponto
            if is_optimal:
                ax.scatter(x1, x2, c='red', s=200, zorder=10, 
                          edgecolors='darkred', linewidth=3)
                point_color = 'red'
            else:
                ax.scatter(x1, x2, c='blue', s=120, zorder=9, 
                          edgecolors='darkblue', linewidth=2)
                point_color = 'blue'
            
            # Label do vértice
            label = vertex_labels[i] if i < len(vertex_labels) else f'P{i}'
            
            # Posicionar label para evitar sobreposição
            offset_x, offset_y = 15, 15
            if x1 > max_x * 0.7:  # Se estiver à direita
                offset_x = -60
            if x2 > max_y * 0.7:  # Se estiver acima
                offset_y = -30
            
            ax.annotate(f'{label} = ({x1:.0f}, {x2:.0f})', 
                       (x1, x2), xytext=(offset_x, offset_y), 
                       textcoords='offset points',
                       fontsize=14, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor=point_color, alpha=0.9, linewidth=2))
            
            # Mostrar valor da função objetivo no ponto ótimo
            if is_optimal:
                obj_value = vertex['objective_value']
                ax.annotate(f'Z = {obj_value:.0f}', 
                           (x1, x2), xytext=(offset_x, offset_y - 25), 
                           textcoords='offset points',
                           fontsize=12, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                    edgecolor='red', alpha=0.9))
        
        # Linha da função objetivo passando pelo ponto ótimo
        if optimal_vertex:
            c1, c2 = objective_coeffs
            opt_value = optimal_vertex['objective_value']
            
            if abs(c2) > self.tolerance:
                # Linha: c1*x1 + c2*x2 = opt_value
                y_obj = (opt_value - c1 * x_line) / c2
                valid_obj = (y_obj >= 0) & (y_obj <= max_y * 1.1) & (x_line >= 0)
                
                if np.any(valid_obj):
                    ax.plot(x_line[valid_obj], y_obj[valid_obj], 'r--', linewidth=4,
                           label=f'Z = {c1}X₁ + {c2}X₂ = {opt_value:.0f}')
        
        # Legenda
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95, 
                 fancybox=True, shadow=True)
        
        # Estilo profissional dos eixos
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        # Ticks mais visíveis
        ax.tick_params(axis='both', which='major', labelsize=12, 
                      width=2, length=6, colors='black')
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.2)
        buffer.seek(0)
        
        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)  # Liberar memória
        
        return graph_base64

# Instanciar solver
solver = SimplexSolver()

# === ROTAS DA API ===

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar se o servidor está funcionando"""
    return jsonify({
        'status': 'OK',
        'message': 'Servidor Python Simplex está ativo',
        'version': '1.0'
    })

@app.route('/api/solve', methods=['POST'])
def solve_simplex():
    """
    Endpoint principal para resolver problemas de programação linear
    
    Formato da requisição:
    {
        "objective": [c1, c2],
        "constraints": [[a11, a12], [a21, a22], ...],
        "bounds": [b1, b2, ...],
        "type": "maximize" ou "minimize"
    }
    """
    
    try:
        data = request.get_json()
        
        # Validar entrada
        if not data:
            return jsonify({
                'success': False,
                'error': 'Dados JSON não fornecidos'
            }), 400
        
        required_fields = ['objective', 'constraints', 'bounds']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo obrigatório ausente: {field}'
                }), 400
        
        objective_coeffs = data['objective']
        constraints = data['constraints']
        constraint_bounds = data['bounds']
        problem_type = data.get('type', 'maximize')
        
        # Validações básicas
        if len(objective_coeffs) != 2:
            return jsonify({
                'success': False,
                'error': 'Função objetivo deve ter exatamente 2 coeficientes'
            }), 400
        
        if len(constraints) != len(constraint_bounds):
            return jsonify({
                'success': False,
                'error': 'Número de restrições não confere com número de limites'
            }), 400
        
        for constraint in constraints:
            if len(constraint) != 2:
                return jsonify({
                    'success': False,
                    'error': 'Cada restrição deve ter exatamente 2 coeficientes'
                }), 400
        
        # Resolver problema
        result = solver.solve_problem(
            objective_coeffs, constraints, constraint_bounds, problem_type
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Retorna exemplos predefinidos"""
    examples = {
        'lucro': {
            'name': 'Maximização de Lucro',
            'objective': [10, 5],
            'constraints': [[1, 1], [2, 1]],
            'bounds': [8, 10],
            'type': 'maximize'
        },
        'producao': {
            'name': 'Mix de Produção',
            'objective': [6, 8],
            'constraints': [[1, 1], [2, 1], [1, 2]],
            'bounds': [7, 10, 8],
            'type': 'maximize'
        },
        'profissional': {
            'name': 'Exemplo Profissional',
            'objective': [35, 80],
            'constraints': [[2, 1], [1, 3]],
            'bounds': [100, 90],
            'type': 'maximize'
        }
    }
    
    return jsonify({
        'success': True,
        'examples': examples
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    