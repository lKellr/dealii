/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Maximilian Bergbauer, Technical University of Munich, 2024
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples.

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

// The first new header contains some common level set functions.
// For example, the spherical geometry that we use here.
#include <deal.II/base/function_signed_distance.h>

// We also need 3 new headers from the NonMatching namespace.
#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

// And most important the header for flexible matrix-free evaluation.
#include <deal.II/matrix_free/fe_point_evaluation.h>

// @sect3{The PoissonSolver class Template}
// We then define the main class that solves the Laplace problem.

namespace Step95
{
  using namespace dealii;

  inline bool is_inside(unsigned int active_fe_index)
  {
    return active_fe_index ==
           (unsigned int)NonMatching::LocationToLevelSet::inside;
  }

  inline bool is_intersected(unsigned int active_fe_index)
  {
    return active_fe_index ==
           (unsigned int)NonMatching::LocationToLevelSet::intersected;
  }

  template <int dim>
  class PoissonOperator
  {
    using Number              = double;
    using VectorizedArrayType = VectorizedArray<Number>;
    using VectorType          = LinearAlgebra::distributed::Vector<double>;
    using CellIntegrator =
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;
    using FaceIntegrator =
      FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;
    using GenericCellIntegrator =
      FEPointEvaluation<1, dim, dim, VectorizedArrayType>;
    using GenericFaceIntegrator =
      FEFacePointEvaluation<1, dim, dim, VectorizedArrayType>;

    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

  public:
    void
    reinit(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free_in,
           const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>
             *mapping_info_cell_in,
           const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>
             *mapping_info_surface_in,
           const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>
                     *mapping_info_faces_in,
           const bool is_dg_in)
    {
      matrix_free          = &matrix_free_in;
      mapping_info_cell    = mapping_info_cell_in;
      mapping_info_surface = mapping_info_surface_in;
      mapping_info_faces   = mapping_info_faces_in;
      is_dg                = is_dg_in;

      evaluator_cell = std::make_unique<GenericCellIntegrator>(
        *mapping_info_cell, matrix_free->get_dof_handler(dof_index).get_fe(0));
      evaluator_surface = std::make_unique<GenericCellIntegrator>(
        *mapping_info_surface,
        matrix_free->get_dof_handler(dof_index).get_fe(0));

      matrix_free->initialize_cell_data_vector(cell_diameter);
      for (unsigned int cell_batch_index = 0;
           cell_batch_index <
           matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
           ++cell_batch_index)
        {
          auto &diameter = cell_diameter[cell_batch_index];
          for (unsigned int v = 0;
               v <
               matrix_free->n_active_entries_per_cell_batch(cell_batch_index);
               ++v)
            {
              const auto cell_accessor_inside =
                matrix_free->get_cell_iterator(cell_batch_index, v);

              diameter[v] = cell_accessor_inside->minimum_vertex_distance();
            }
        }
    }

    void vmult(VectorType &dst, const VectorType &src) const
    {
      matrix_free->loop(&PoissonOperator::local_apply_cell,
                        &PoissonOperator::local_apply_face,
                        &PoissonOperator::local_apply_boundary_face,
                        this,
                        dst,
                        src,
                        true);
    }

    void rhs(VectorType &rhs, const Function<dim> &rhs_function)
    {
      // TODO
    }

  private:
    void local_apply_cell(
      const MatrixFree<dim, Number, VectorizedArrayType> &,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      // TODO
    }

    void local_apply_face(
      const MatrixFree<dim, Number, VectorizedArrayType> &,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // TODO
    }

    void local_apply_boundary_face(
      const MatrixFree<dim, Number, VectorizedArrayType> &,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // TODO
    }

    inline bool
    is_inside_face(std::pair<unsigned int, unsigned int> face_category) const
    {
      return is_inside(face_category.first) && is_inside(face_category.second);
    }

    inline bool
    is_mixed_face(std::pair<unsigned int, unsigned int> face_category) const
    {
      return (is_inside(face_category.first) &&
              is_intersected(face_category.second)) ||
             (is_intersected(face_category.first) &&
              is_inside(face_category.second));
    }

    inline bool is_intersected_face(
      std::pair<unsigned int, unsigned int> face_category) const
    {
      return is_intersected(face_category.first) &&
             is_intersected(face_category.second);
    }

    VectorizedArrayType
    compute_diameter_of_inner_face_batch(unsigned int face_batch_index) const
    {
      const auto &face_info = matrix_free->get_face_info(face_batch_index);

      VectorizedArrayType diameter = 0.;
      for (unsigned int v = 0;
           v < matrix_free->n_active_entries_per_face_batch(face_batch_index);
           ++v)
        {
          const auto cell_batch_index_interior =
            face_info.cells_interior[v] / n_lanes;
          const auto cell_lane_index_interior =
            face_info.cells_interior[v] % n_lanes;
          const auto cell_batch_index_exterior =
            face_info.cells_exterior[v] / n_lanes;
          const auto cell_lane_index_exterior =
            face_info.cells_exterior[v] % n_lanes;

          diameter[v] = std::max(
            cell_diameter[cell_batch_index_interior][cell_lane_index_interior],
            cell_diameter[cell_batch_index_exterior][cell_lane_index_exterior]);
        }

      return diameter;
    }

    VectorizedArrayType
    compute_diameter_of_boundary_face_batch(unsigned int face_batch_index) const
    {
      const auto &face_info = matrix_free->get_face_info(face_batch_index);

      VectorizedArrayType diameter = 0.;
      for (unsigned int v = 0;
           v < matrix_free->n_active_entries_per_face_batch(face_batch_index);
           ++v)
        {
          const auto cell_batch_index_interior =
            face_info.cells_interior[v] / n_lanes;
          const auto cell_lane_index_interior =
            face_info.cells_interior[v] % n_lanes;

          diameter[v] =
            cell_diameter[cell_batch_index_interior][cell_lane_index_interior];
        }

      return diameter;
    }

    ObserverPointer<const MatrixFree<dim, Number, VectorizedArrayType>>
      matrix_free;
    ObserverPointer<
      const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_cell;
    ObserverPointer<
      const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_surface;
    ObserverPointer<
      const NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_faces;

    std::unique_ptr<GenericCellIntegrator> evaluator_cell;
    std::unique_ptr<GenericCellIntegrator> evaluator_surface;

    AlignedVector<VectorizedArrayType> cell_diameter;

    const unsigned int dof_index  = 0;
    const unsigned int quad_index = 0;

    bool is_dg = false;
  };

  template <int dim>
  class PoissonSolver
  {
    using Number              = double;
    using VectorizedArrayType = VectorizedArray<Number>;
    using VectorType          = LinearAlgebra::distributed::Vector<double>;

    using CellIntegrator =
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;
    using GenericCellIntegrator =
      FEPointEvaluation<1, dim, dim, VectorizedArrayType>;

    static constexpr unsigned int n_lanes = VectorizedArrayType::size();

  public:
    PoissonSolver();

    void run();

  private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void setup_mapping_data();

    void solve();

    void output_results() const;

    double compute_L2_error() const;

    const unsigned int fe_degree;

    const Functions::ConstantFunction<dim> rhs_function;
    const Functions::ConstantFunction<dim> boundary_condition;

    parallel::distributed::Triangulation<dim> triangulation;

    ConditionalOStream pcout;

    MatrixFree<dim, double> matrix_free;

    PoissonOperator<dim> poisson_operator;

    // We need two separate DoFHandlers. The first manages the DoFs for the
    // discrete level set function that describes the geometry of the domain.
    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    VectorType      level_set;

    // The second DoFHandler manages the DoFs for the solution of the Poisson
    // equation.
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim>       dof_handler;
    VectorType            solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    VectorType rhs;

    const MappingQ<dim> mapping;

    // mapping info objects
    std::unique_ptr<NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_cell;
    std::unique_ptr<NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_surface;
    std::unique_ptr<NonMatching::MappingInfo<dim, dim, VectorizedArrayType>>
      mapping_info_faces;

    const unsigned int dof_index  = 0;
    const unsigned int quad_index = 0;

    const bool is_dg = false;
  };



  template <int dim>
  PoissonSolver<dim>::PoissonSolver()
    : fe_degree(1)
    , rhs_function(4.0)
    , boundary_condition(1.0)
    , triangulation(MPI_COMM_WORLD)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(
              triangulation.get_communicator()) == 0)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , mesh_classifier(level_set_dof_handler, level_set)
    , mapping(1)
  {}



  // @sect3{Setting up the Background Mesh}
  // We generate a background mesh with perfectly Cartesian cells. Our domain is
  // a unit disc centered at the origin, so we need to make the background mesh
  // a bit larger than $[-1, 1]^{\text{dim}}$ to completely cover $\Omega$.
  template <int dim>
  void PoissonSolver<dim>::make_grid()
  {
    pcout << "Creating background mesh" << std::endl;

    GridGenerator::hyper_cube(triangulation, -1.21, 1.21);
    triangulation.refine_global(2);
  }



  // @sect3{Setting up the Discrete Level Set Function}
  // The discrete level set function is defined on the whole background mesh.
  // Thus, to set up the DoFHandler for the level set function, we distribute
  // DoFs over all elements in $\mathcal{T}_h$. We then set up the discrete
  // level set function by interpolating onto this finite element space.
  template <int dim>
  void PoissonSolver<dim>::setup_discrete_level_set()
  {
    pcout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
  }



  // We then use the MeshClassifier to check LocationToLevelSet for each cell in
  // the mesh and tell the DoFHandler to use FE_Q on elements that are inside or
  // intersected, and FE_Nothing on the elements that are outside.
  template <int dim>
  void PoissonSolver<dim>::distribute_dofs()
  {
    pcout << "Distributing degrees of freedom" << std::endl;

    fe_collection.push_back(FE_Q<dim>(fe_degree)); // inside
    fe_collection.push_back(FE_Nothing<dim>());    // outside
    fe_collection.push_back(FE_Q<dim>(fe_degree)); // intersected

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::inside)
          cell->set_active_fe_index(
            (unsigned int)NonMatching::LocationToLevelSet::inside);
        else if (cell_location == NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(
            (unsigned int)NonMatching::LocationToLevelSet::outside);
        else if (cell_location == NonMatching::LocationToLevelSet::intersected)
          cell->set_active_fe_index(
            (unsigned int)NonMatching::LocationToLevelSet::intersected);
        else
          cell->set_active_fe_index(
            (unsigned int)NonMatching::LocationToLevelSet::outside);
      }

    dof_handler.distribute_dofs(fe_collection);
  }



  template <int dim>
  void PoissonSolver<dim>::setup_mapping_data()
  {
    auto is_intersected_cell =
      [&](const TriaIterator<CellAccessor<dim, dim>> &cell) {
        return mesh_classifier.location_to_level_set(cell) ==
               NonMatching::LocationToLevelSet::intersected;
      };

    std::vector<Quadrature<dim>> quad_vec_cells;
    quad_vec_cells.reserve(
      (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
      n_lanes);

    std::vector<NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    quad_vec_surface.reserve(
      (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
      n_lanes * n_lanes);

    hp::QCollection<1> q_collection1D(QGauss<1>(fe_degree + 1));

    NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection1D, level_set_dof_handler, level_set);

    std::vector<typename DoFHandler<dim>::cell_iterator> vector_accessors;
    vector_accessors.reserve(
      (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
      n_lanes);
    for (unsigned int cell_batch = 0;
         cell_batch <
         matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
         ++cell_batch)
      for (unsigned int v = 0; v < n_lanes; ++v)
        {
          if (v < matrix_free.n_active_entries_per_cell_batch(cell_batch))
            vector_accessors.push_back(
              matrix_free.get_cell_iterator(cell_batch, v));
          else
            vector_accessors.push_back(
              matrix_free.get_cell_iterator(cell_batch, 0));

          const auto &cell = vector_accessors.back();

          if (is_intersected_cell(cell))
            {
              quadrature_generator.generate(cell);

              quad_vec_cells.push_back(
                quadrature_generator.get_inside_quadrature());
              quad_vec_surface.push_back(
                quadrature_generator.get_surface_quadrature());
            }
          else
            {
              quad_vec_cells.emplace_back();
              quad_vec_surface.emplace_back();
            }
        }

    // initialize MappingInfo objects to precompute mapping information
    mapping_info_cell = std::make_unique<
      NonMatching::MappingInfo<dim, dim, VectorizedArray<Number>>>(
      mapping, update_values | update_gradients | update_JxW_values);
    mapping_info_cell->reinit_cells(vector_accessors, quad_vec_cells);

    mapping_info_surface = std::make_unique<
      NonMatching::MappingInfo<dim, dim, VectorizedArray<Number>>>(
      mapping,
      update_values | update_gradients | update_JxW_values |
        update_normal_vectors);
    mapping_info_surface->reinit_surface(vector_accessors, quad_vec_surface);

    // faces
    if (is_dg)
      {
        NonMatching::DiscreteFaceQuadratureGenerator<dim>
          face_quadrature_generator(q_collection1D,
                                    level_set_dof_handler,
                                    level_set);

        std::vector<Quadrature<dim - 1>> quad_vec_faces;
        quad_vec_faces.reserve((matrix_free.n_inner_face_batches() +
                                matrix_free.n_boundary_face_batches() +
                                matrix_free.n_ghost_inner_face_batches()) *
                               n_lanes);
        std::vector<
          std::pair<typename DoFHandler<dim>::cell_iterator, unsigned int>>
          vector_face_accessors_m;
        vector_face_accessors_m.reserve(
          (matrix_free.n_inner_face_batches() +
           matrix_free.n_boundary_face_batches() +
           matrix_free.n_ghost_inner_face_batches()) *
          n_lanes);
        // fill container for inner face batches
        unsigned int face_batch = 0;
        for (; face_batch < matrix_free.n_inner_face_batches(); ++face_batch)
          {
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                if (v < matrix_free.n_active_entries_per_face_batch(face_batch))
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, v, true));
                else
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, 0, true));

                const auto &cell_m = vector_face_accessors_m.back().first;

                const unsigned int f = vector_face_accessors_m.back().second;

                if (is_intersected_cell(cell_m))
                  {
                    face_quadrature_generator.generate(cell_m, f);
                    quad_vec_faces.push_back(
                      face_quadrature_generator.get_inside_quadrature());
                  }
                else
                  quad_vec_faces.emplace_back();
              }
          }
        // and boundary face batches
        for (; face_batch < (matrix_free.n_inner_face_batches() +
                             matrix_free.n_boundary_face_batches());
             ++face_batch)
          {
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                if (v < matrix_free.n_active_entries_per_face_batch(face_batch))
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, v, true));
                else
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, 0, true));

                const auto &cell_m = vector_face_accessors_m.back().first;

                const unsigned int f = vector_face_accessors_m.back().second;

                if (is_intersected_cell(cell_m))
                  {
                    face_quadrature_generator.generate(cell_m, f);
                    quad_vec_faces.push_back(
                      face_quadrature_generator.get_inside_quadrature());
                  }
                else
                  quad_vec_faces.emplace_back();
              }
          }
        for (; face_batch < (matrix_free.n_inner_face_batches() +
                             matrix_free.n_boundary_face_batches() +
                             matrix_free.n_ghost_inner_face_batches());
             ++face_batch)
          {
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                if (v < matrix_free.n_active_entries_per_face_batch(face_batch))
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, v, true));
                else
                  vector_face_accessors_m.push_back(
                    matrix_free.get_face_iterator(face_batch, 0, true));

                const auto &cell_m = vector_face_accessors_m.back().first;

                const unsigned int f = vector_face_accessors_m.back().second;

                if (is_intersected_cell(cell_m))
                  {
                    face_quadrature_generator.generate(cell_m, f);
                    quad_vec_faces.push_back(
                      face_quadrature_generator.get_inside_quadrature());
                  }
                else
                  quad_vec_faces.emplace_back();
              }
          }

        mapping_info_faces = std::make_unique<
          NonMatching::MappingInfo<dim, dim, VectorizedArray<Number>>>(
          mapping,
          update_values | update_gradients | update_JxW_values |
            update_normal_vectors);
        mapping_info_faces->reinit_faces(vector_face_accessors_m,
                                         quad_vec_faces);
      }
  }


  // @sect3{Solving the System}
  template <int dim>
  void PoissonSolver<dim>::solve()
  {
    pcout << "Solving system" << std::endl;

    const unsigned int   max_iterations = solution.size();
    SolverControl        solver_control(max_iterations);
    SolverCG<VectorType> solver(solver_control);
    solver.solve(poisson_operator, solution, rhs, PreconditionIdentity());
  }



  // @sect3{Data Output}
  // Since both DoFHandler instances use the same triangulation, we can add both
  // the level set function and the solution to the same vtu-file. Further, we
  // do not want to output the cells that have LocationToLevelSet value outside.
  // To disregard them, we write a small lambda function and use the
  // set_cell_selection function of the DataOut class.
  template <int dim>
  void PoissonSolver<dim>::output_results() const
  {
    pcout << "Writing vtu file" << std::endl;

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() && cell->is_locally_owned() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./",
                                        "step-95",
                                        0,
                                        triangulation.get_communicator());
  }



  // @sect3{L2-Error}
  // To test that the implementation works as expected, we want to compute the
  // error in the solution in the $L^2$-norm. The analytical solution to the
  // Poisson problem stated in the introduction reads
  // @f{align*}{
  //  u(x) = 1 - \frac{2}{\text{dim}}(\| x \|^2 - 1) , \qquad x \in
  //  \overline{\Omega}.
  // @f}
  // We first create a function corresponding to the analytical solution:
  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim>  &point,
                 const unsigned int component = 0) const override;
  };



  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim>  &point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return -2. / dim * (point.norm_square() - 1.);
  }



  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.
  template <int dim>
  double PoissonSolver<dim>::compute_L2_error() const
  {
    pcout << "Computing L2 error" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    // We then iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    AnalyticalSolution<dim> analytical_solution;
    double                  error_L2_squared = 0;

    auto l2_kernel = [](auto                &evaluator,
                        const Function<dim> &analytical_solution_function,
                        const unsigned int   q) {
      const auto q_points = evaluator.quadrature_point(q);

      VectorizedArrayType value = 0.;

      for (unsigned int v = 0; v < n_lanes; ++v)
        {
          Point<dim> q_point;
          for (unsigned int d = 0; d < dim; ++d)
            q_point[d] = q_points[d][v];

          value[v] = analytical_solution_function.value(q_point);
        }

      const auto difference = evaluator.get_value(q) - value;

      evaluator.submit_value(difference * difference, q);
    };

    // matrix-free loop
    unsigned int dummy = 0;
    matrix_free.template cell_loop<unsigned int, VectorType>(
      [&](const MatrixFree<dim, Number, VectorizedArrayType> &,
          unsigned int &,
          const VectorType                            &src,
          const std::pair<unsigned int, unsigned int> &cell_range) {
        CellIntegrator evaluator(matrix_free, dof_index, quad_index);

        GenericCellIntegrator evaluator_cell(
          *mapping_info_cell, matrix_free.get_dof_handler(dof_index).get_fe(0));

        const auto cell_range_category =
          matrix_free.get_cell_range_category(cell_range);

        if (is_inside(cell_range_category))
          {
            for (unsigned int cell_batch_index = cell_range.first;
                 cell_batch_index < cell_range.second;
                 ++cell_batch_index)
              {
                evaluator.reinit(cell_batch_index);

                evaluator.read_dof_values(src);

                evaluator.evaluate(EvaluationFlags::values);

                for (unsigned int q : evaluator.quadrature_point_indices())
                  l2_kernel(evaluator, analytical_solution, q);

                for (unsigned int v = 0;
                     v < matrix_free.n_active_entries_per_cell_batch(
                           cell_batch_index);
                     ++v)
                  error_L2_squared += evaluator.integrate_value()[v];
              }
          }
        else if (is_intersected(cell_range_category))
          {
            const auto dofs_per_cell = evaluator.dofs_per_cell;

            for (unsigned int cell_batch_index = cell_range.first;
                 cell_batch_index < cell_range.second;
                 ++cell_batch_index)
              {
                evaluator.reinit(cell_batch_index);
                evaluator.read_dof_values(src);

                for (unsigned int v = 0;
                     v < matrix_free.n_active_entries_per_cell_batch(
                           cell_batch_index);
                     ++v)
                  {
                    const unsigned int cell_index =
                      cell_batch_index * n_lanes + v;

                    evaluator_cell.reinit(cell_index);

                    evaluator_cell.evaluate(
                      StridedArrayView<const Number, n_lanes>(
                        &evaluator.begin_dof_values()[0][v], dofs_per_cell),
                      EvaluationFlags::values);

                    for (const auto q :
                         evaluator_cell.quadrature_point_indices())
                      l2_kernel(evaluator_cell, analytical_solution, q);

                    error_L2_squared += evaluator_cell.integrate_value();
                  }
              }
          }
      },
      dummy,
      solution);

    return std::sqrt(
      Utilities::MPI::sum(error_L2_squared, triangulation.get_communicator()));
  }



  // @sect3{A Convergence Study}
  // Finally, we do a convergence study to check that the $L^2$-error decreases
  // with the expected rate. We refine the background mesh a few times. In each
  // refinement cycle, we solve the problem, compute the error, and add the
  // $L^2$-error and the mesh size to a ConvergenceTable.
  template <int dim>
  void PoissonSolver<dim>::run()
  {
    dealii::Timer      timer;
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 3;

    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        pcout << "Refinement cycle " << cycle << std::endl;
        triangulation.refine_global(1);
        setup_discrete_level_set();
        pcout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
        distribute_dofs();

        QGauss<1>         quadrature(fe_degree + 1);
        AffineConstraints affine_constraints;
        affine_constraints.close();

        // setup matrixfree additional data
        typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
          additional_data;
        additional_data.mapping_update_flags =
          update_gradients | update_values | update_quadrature_points;
        additional_data.mapping_update_flags_inner_faces =
          update_values | update_gradients;
        additional_data.mapping_update_flags_boundary_faces =
          update_values | update_gradients | update_quadrature_points;
        if (dof_handler.get_fe().degree > 1)
          additional_data.mapping_update_flags_inner_faces |= update_hessians;

        // setup matrixfree object
        matrix_free.reinit(mapping,
                           dof_handler,
                           affine_constraints,
                           quadrature,
                           additional_data);

        setup_mapping_data();

        poisson_operator.reinit(matrix_free,
                                mapping_info_cell.get(),
                                mapping_info_surface.get(),
                                mapping_info_faces.get(),
                                is_dg);

        matrix_free.initialize_dof_vector(solution);
        matrix_free.initialize_dof_vector(rhs);

        poisson_operator.rhs(rhs, rhs_function);

        solve();
        if (cycle == 3)
          output_results();
        const double error_L2 = compute_L2_error();
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("Mesh size", cell_side_length);
        convergence_table.add_value("L2-Error", error_L2);

        convergence_table.evaluate_convergence_rates(
          "L2-Error", ConvergenceTable::reduction_rate_log2);
        convergence_table.set_scientific("L2-Error", true);

        pcout << std::endl;
        if (Utilities::MPI::this_mpi_process(
              triangulation.get_communicator()) == 0)
          convergence_table.write_text(pcout.get_stream());
        pcout << std::endl;
      }
    pcout << "wall time: " << timer.wall_time() << std::endl;
  }

} // namespace Step95



// @sect3{The main() function}
int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  constexpr int dim = 3;

  Step95::PoissonSolver<dim> poisson_solver;
  poisson_solver.run();
}
