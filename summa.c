#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/************************************************************/
/*Imprimix una matriu                                       */
/************************************************************/
void imprimir_matriu(int grau_matriu, double *matriu) {
  int j, i;
  for(i = 0; i < grau_matriu; i++) {
    for(j = 0; j < grau_matriu; j++) {
      printf("%.2f ", matriu[i * grau_matriu + j]);
    }
    printf("\n");
  }
}

/************************************************************/
/*Multiplica A i B i la guarda en C                         */
/************************************************************/
void matriu_mult(int grau_matriu, double *A, double *B, double*C) {
  memset(C, 0.0, grau_matriu*grau_matriu*sizeof(double));
  int i, j, l;
  for (i = 0; i < grau_matriu; i++) {
    for (j = 0; j < grau_matriu; j++) {
      for (l = 0; l < grau_matriu; l++) {
        C[i*grau_matriu+j] += A[i*grau_matriu+l]*B[l*grau_matriu+j];
      }
    }
  }
}

/************************************************************/
/*Compara dos matrius                                       */
/************************************************************/
int compara_matrius(int grau_matriu, double *A, double *B) {
  int i;
  for(i=0; i<grau_matriu*grau_matriu; i++) {
    if(A[i] != B[i]) { return 0; }
  }
  return 1;
}

/************************************************************/
/*Plena la matriu am nombre aleatoris                       */
/************************************************************/
void plena_matriu(int grau_matriu, double *matriu) {
  int j, i;
  for(i = 0; i<grau_matriu; i++) {
    for(j = 0; j<grau_matriu; j++) {
      matriu[i*grau_matriu + j] = (((double) rand()) / RAND_MAX) + rand()%11;;
    }
  }
}

/************************************************************/
/*Funcion per a repartir les dades                          */
/************************************************************/
void funcio_scatter(int grau_b, int grau_m, int num_p, double *matriu, double *matriu_local) {
  int *elems_per_proc = (int *) malloc(num_p*sizeof(int));
  int *despl_send_buff = (int *) malloc(num_p*sizeof(int));
  int sqr_nump = (int)(round(sqrt(num_p)));

  int i, j;
  for (i = 0; i < sqr_nump; i++) {
    for (j = 0; j < sqr_nump; j++) {
      despl_send_buff[i*sqr_nump+j] = i*grau_m*grau_b+j*grau_b;
      elems_per_proc[i*sqr_nump+j] = 1;
    }
  }

  MPI_Datatype datatype, temp;
  MPI_Type_vector(grau_b, grau_b, grau_m, MPI_DOUBLE, &temp);
  MPI_Type_create_resized(temp, 0, sizeof(double), &datatype);
  MPI_Type_commit(&datatype);

  MPI_Scatterv(matriu, elems_per_proc, despl_send_buff, datatype, matriu_local, grau_b*grau_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Type_free(&datatype);
  MPI_Type_free(&temp);
}

/************************************************************/
/*Funcion per a recollir les dades                          */
/************************************************************/
void funcio_gather(int grau_b, int grau_m, int num_p, double *matriu, double *matriu_local) {
  int *elems_per_proc = (int *) malloc(num_p*sizeof(int));
  int *despl_send_buff = (int *) malloc(num_p*sizeof(int));
  int sqr_nump = (int)(round(sqrt(num_p)));

  int i, j;
  for (i = 0; i < sqr_nump; i++) {
    for (j = 0; j < sqr_nump; j++) {
      despl_send_buff[i*sqr_nump+j] = i*grau_m*grau_b+j*grau_b;
      elems_per_proc[i*sqr_nump+j] = 1;
    }
  }

  MPI_Datatype datatype, temp;
  MPI_Type_vector(grau_b, grau_b, grau_m, MPI_DOUBLE, &temp);
  MPI_Type_create_resized(temp, 0, sizeof(double), &datatype);
  MPI_Type_commit(&datatype);

  MPI_Gatherv(matriu_local, grau_b*grau_b, MPI_DOUBLE, matriu, elems_per_proc, despl_send_buff, datatype, 0, MPI_COMM_WORLD);

  MPI_Type_free(&datatype);
  MPI_Type_free(&temp);
}

int main( int argc, char *argv[] ) {
  // Grau de les matrius, ID del procesador i numero de procesadors
  int grau_matriu, rank, num_procs;
  // Variables generals
  int i, j;
  if(argc>1) {
    grau_matriu = atoi(argv[1]);
  } else {
    grau_matriu = 2;
  }
  // Iniciar MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  // Per a comprovacions i repartiment en blocs
  int sqr_numprocs = (int)(round(sqrt(num_procs)));

  /************************************************************/
  /*Comprovar requisit de execució                            */
  /************************************************************/
  if(grau_matriu%sqr_numprocs) {
    if(rank == 0) {
      printf("El grau de la matriu ha de ser divisible por l'arrel del numero de procesadors.\n");
    }
    MPI_Finalize();
    return 0;
  }

  /************************************************************/
  /*Calcule els blocs per procesador                          */
  /************************************************************/
  int grau_bloc = grau_matriu / sqr_numprocs;

  /************************************************************/
  /*Rreserve memoria a gastar                                 */
  /************************************************************/
  double *A, *B, *C, *C_comprovant;
  double *A_local = (double *) malloc(grau_bloc*grau_bloc*sizeof(double));
  double *B_local = (double *) malloc(grau_bloc*grau_bloc*sizeof(double));
  double *C_local = (double *) malloc(grau_bloc*grau_bloc*sizeof(double));
  double *A_aux = (double *) malloc(grau_bloc*grau_bloc*sizeof(double));
  double *B_aux = (double *) malloc(grau_bloc*grau_bloc*sizeof(double));
  // El proces 0 inicialitza les matrius A, B i C junt amb la de comprovació C_comprovant
  if(rank == 0) {
    A = (double *) malloc(grau_matriu*grau_matriu*sizeof(double));
    B = (double *) malloc(grau_matriu*grau_matriu*sizeof(double));
    C = (double *) malloc(grau_matriu*grau_matriu*sizeof(double));
    C_comprovant = (double *) malloc(grau_matriu*grau_matriu*sizeof(double));
    // Plene A i B
    plena_matriu(grau_matriu, A);
    plena_matriu(grau_matriu, B);
    // Plene C i C_comprovant anb 0s
    memset(C, 0.0, grau_matriu*grau_matriu*sizeof(double));
    memset(C_comprovant, 0.0, grau_matriu*grau_matriu*sizeof(double));
    // Si el grau no es massa gran les imprimisc
    if(grau_matriu <= 10) {
      printf("\nA:\n");
      imprimir_matriu(grau_matriu, A);
      printf("\nB:\n");
      imprimir_matriu(grau_matriu, B);
    }
  }

  /************************************************************/
  /*Calcul de temps inicial                                   */
  /************************************************************/
  MPI_Barrier(MPI_COMM_WORLD);
  double t_ini = MPI_Wtime();

  /************************************************************/
  /*Realitzem el scatter de les dades                         */
  /************************************************************/
  funcio_scatter(grau_bloc, grau_matriu, num_procs, A, A_local);
  funcio_scatter(grau_bloc, grau_matriu, num_procs, B, B_local);

  /************************************************************/
  /*Creacio dels comunicadors                                 */
  /************************************************************/
  int ndims = 2;
  int dims_size[2] = {sqr_numprocs, sqr_numprocs};
  int periods[2] = {0, 0};
  int reorder = 0;
  MPI_Comm comm_card;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_size, periods, reorder, &comm_card);
  int coords[2];
  MPI_Cart_coords(comm_card, rank, 2, coords);

  MPI_Comm comm_row, comm_col;
  int belongs[2];
  belongs[0] = 0;
  belongs[1] = 1;
  MPI_Cart_sub(comm_card, belongs, &comm_row);
  belongs[0] = 1;
  belongs[1] = 0;
  MPI_Cart_sub(comm_card, belongs, &comm_col);

  /************************************************************/
  /*Algorime SUMMA                                            */
  /************************************************************/
  int pas;
  for (pas = 0; pas < sqr_numprocs; pas++) {
    double *C_aux = malloc(grau_bloc*grau_bloc*sizeof(double));
    if (pas == coords[1]) {
      for (i = 0; i < grau_bloc; i++) {
        for (j = 0; j < grau_bloc; j++) {
          A_aux[i*grau_bloc+j] = A_local[i*grau_bloc+j];
        }
      }
    }
    MPI_Bcast(A_aux, grau_bloc*grau_bloc, MPI_DOUBLE, pas, comm_row);

    if (pas == coords[0]) {
      for (i = 0; i < grau_bloc; i++) {
        for (j = 0; j < grau_bloc; j++) {
          B_aux[i*grau_bloc+j] = B_local[i*grau_bloc+j];
        }
      }
    }
    MPI_Bcast(B_aux, grau_bloc*grau_bloc, MPI_DOUBLE, pas, comm_col);

    matriu_mult(grau_bloc, A_aux, B_aux, C_aux);

    for(i = 0; i < grau_bloc*grau_bloc; i++) { C_local[i] += C_aux[i]; }

    free(C_aux);
  }

  /************************************************************/
  /*Recollim totes les dades de les C locals en la C final    */
  /************************************************************/
  funcio_gather(grau_bloc, grau_matriu, num_procs, C, C_local);

  /************************************************************/
  /*Calcul de temps final i total                             */
  /************************************************************/
  MPI_Barrier(MPI_COMM_WORLD);
  double t_fin = MPI_Wtime();
  double t_total = t_fin - t_ini;

  /************************************************************/
  /*Mire les matrius locals                                   */
  /************************************************************/
  if(grau_matriu <= 0) { // Si el grau no es massa gran les imprimisc
    for(i = 0; i < num_procs; i++) {
      if(i == rank) {
        printf("\nSoc proc %d amb A local:\n", i);
        imprimir_matriu(grau_bloc, A_local);
        printf("\nSoc proc %d amb B local:\n", i);
        imprimir_matriu(grau_bloc, B_local);
        printf("\nSoc proc %d amb C local:\n", i);
        imprimir_matriu(grau_bloc, C_local);
      }
    }
  }

  /************************************************************/
  /*Imprisc el resulta i el temps d'execucio                  */
  /************************************************************/
  if(rank == 0) {
    if(grau_matriu <= 10) { // Si el grau no es massa gran la imprimisc
      printf("\nC:\n");
      imprimir_matriu(grau_matriu, C);
    }
    printf("\nEl algoritme SUMMA ha tardat %f seg\n", t_total);
    // Clacule la mateixa matriu pero de manera secuencial per a comparar els temps
    t_ini = MPI_Wtime();
    matriu_mult(grau_matriu, A, B, C_comprovant);
    t_fin = MPI_Wtime();
    t_total = t_fin - t_ini;

    if(grau_matriu <= 10) { // Si el grau no es massa gran la imprimisc
      printf("\nC_comprovant:\n");
      imprimir_matriu(grau_matriu, C_comprovant);
    }
    printf("\nEl algoritme secuencial de multiplicacio ha tardat %f seg\n", t_total);
    // Me asegure de que son iguals
    if(compara_matrius(grau_matriu, C, C_comprovant) == 1) {
      printf("\nCalcul correcte!\n");
    } else { printf("\nCalcul erroni!\n"); }

    // Alliverar espai
    free(A);
    free(B);
    free(C);
    free(C_comprovant);
  }
  // Alliverar espai
  MPI_Comm_free(&comm_card);
  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_col);
  free(A_aux);
  free(B_aux);
  free(A_local);
  free(B_local);
  free(C_local);

  MPI_Finalize();
  return 0;
}
