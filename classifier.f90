SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
!
!   -- Reference BLAS level3 routine --
!   -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
!   -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
!
!   .. Scalar Arguments ..
    DOUBLE PRECISION ALPHA,BETA
    INTEGER K,LDA,LDB,LDC,M,N
    CHARACTER TRANSA,TRANSB
!   .. Array Arguments ..
    DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
!   .. Local Scalars ..
    DOUBLE PRECISION TEMP
    INTEGER I,INFO,J,L,NROWA,NROWB
    LOGICAL NOTA,NOTB
!   .. Parameters ..
    DOUBLE PRECISION ONE,ZERO
    PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
    INTRINSIC MAX
!
!   Set NOTA and NOTB as true if A and B respectively are not transposed
!   and set NROWA and NROWB as the number of rows of A and B respectively.
	NOTA = TRANSA == 'N'
	NOTB = TRANSB == 'N'
    IF (NOTA) THEN
        NROWA = M
    ELSE
        NROWA = K
    END IF
    IF (NOTB) THEN
        NROWB = K
    ELSE
        NROWB = N
    END IF
!
!   Test the input parameters.
    INFO = 0
    IF ((.NOT.NOTA) .AND. (TRANSA /= 'C') .AND. (TRANSA /= 'T')) THEN
        INFO = 1
    ELSE IF ((.NOT.NOTB) .AND. (TRANSB /= 'C') .AND. (TRANSB /= 'T')) THEN
        INFO = 2
    ELSE IF (M.LT.0) THEN
        INFO = 3
    ELSE IF (N.LT.0) THEN
        INFO = 4
    ELSE IF (K.LT.0) THEN
        INFO = 5
    ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
        INFO = 8
    ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
        INFO = 10
    ELSE IF (LDC.LT.MAX(1,M)) THEN
        INFO = 13
    END IF
    IF (INFO.NE.0) THEN
        PRINT *, "DGEMM: Error occurred, INFO ", INFO
        RETURN
    END IF
!
!   Quick return if possible.
    IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (((ALPHA.EQ.ZERO).OR.(K.EQ.0)).AND.(BETA.EQ.ONE))) RETURN
!
!   And if alpha.eq.zero.
    IF (ALPHA.EQ.ZERO) THEN
        IF (BETA.EQ.ZERO) THEN
            DO 20 J = 1,N
                DO 10 I = 1,M
                    C(I,J) = ZERO
10              CONTINUE
20          CONTINUE
        ELSE
            DO 40 J = 1,N
                DO 30 I = 1,M
                    C(I,J) = BETA*C(I,J)
30              CONTINUE
40          CONTINUE
        END IF
        RETURN
    END IF
!
!   Start the operations.
    IF (NOTB) THEN
        IF (NOTA) THEN
!
!         Form  C := alpha*A*B + beta*C
            DO 90 J = 1,N
                IF (BETA.EQ.ZERO) THEN
                    DO 50 I = 1,M
                        C(I,J) = ZERO
50                  CONTINUE
                ELSE IF (BETA.NE.ONE) THEN
                    DO 60 I = 1,M
                        C(I,J) = BETA*C(I,J)
60                  CONTINUE
                END IF
                DO 80 L = 1,K
                    TEMP = ALPHA*B(L,J)
                    DO 70 I = 1,M
                        C(I,J) = C(I,J) + TEMP*A(I,L)
70                 CONTINUE
80              CONTINUE
90          CONTINUE
        ELSE
!
!         Form  C := alpha*A**T*B + beta*C
            DO 120 J = 1,N
                DO 110 I = 1,M
                    TEMP = ZERO
                    DO 100 L = 1,K
                        TEMP = TEMP + A(L,I)*B(L,J)
100                 CONTINUE
                    IF (BETA.EQ.ZERO) THEN
                        C(I,J) = ALPHA*TEMP
                    ELSE
                        C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                    END IF
110             CONTINUE
120         CONTINUE
        END IF
    ELSE
        IF (NOTA) THEN
!
!         Form  C := alpha*A*B**T + beta*C
            DO 170 J = 1,N
                IF (BETA.EQ.ZERO) THEN
                    DO 130 I = 1,M
                        C(I,J) = ZERO
130                 CONTINUE
                ELSE IF (BETA.NE.ONE) THEN
                    DO 140 I = 1,M
                        C(I,J) = BETA*C(I,J)
140                 CONTINUE
                END IF
                DO 160 L = 1,K
                    TEMP = ALPHA*B(J,L)
                    DO 150 I = 1,M
                        C(I,J) = C(I,J) + TEMP*A(I,L)
150                 CONTINUE
160             CONTINUE
170         CONTINUE
        ELSE
!
!         Form  C := alpha*A**T*B**T + beta*C
            DO 200 J = 1,N
                DO 190 I = 1,M
                    TEMP = ZERO
                    DO 180 L = 1,K
                        TEMP = TEMP + A(L,I)*B(J,L)
180                 CONTINUE
                    IF (BETA.EQ.ZERO) THEN
                        C(I,J) = ALPHA*TEMP
                    ELSE
                        C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                    END IF
190             CONTINUE
200         CONTINUE
        END IF
    END IF
!
    RETURN
!
!   End of DGEMM
END

  
SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
    DOUBLE PRECISION ALPHA,BETA
    INTEGER INCX,INCY,LDA,M,N
    CHARACTER TRANS
    DOUBLE PRECISION A(LDA,*),X(*),Y(*)
    DOUBLE PRECISION ONE,ZERO
    PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
    DOUBLE PRECISION TEMP
    INTEGER I,INFO,IX,IY,J,JX,JY,KX,KY,LENX,LENY
    INTRINSIC MAX
    INFO = 0
    IF ((TRANS /= 'N') .AND. (TRANS /= 'T') .AND. (TRANS /= 'C')) THEN
        INFO = 1
    ELSE IF (M.LT.0) THEN
        INFO = 2
    ELSE IF (N.LT.0) THEN
        INFO = 3
    ELSE IF (LDA.LT.MAX(1,M)) THEN
        INFO = 6
    ELSE IF (INCX.EQ.0) THEN
        INFO = 8
    ELSE IF (INCY.EQ.0) THEN
        INFO = 11
    END IF
    IF (INFO.NE.0) THEN
        PRINT *, "DGEMV: Error occured, INFO ", INFO
        RETURN
    END IF
    IF ((M.EQ.0) .OR. (N.EQ.0) .OR. ((ALPHA.EQ.ZERO).AND. (BETA.EQ.ONE))) RETURN
    IF (TRANS == 'N') THEN
        LENX = N
        LENY = M
    ELSE
        LENX = M
        LENY = N
    END IF
    IF (INCX.GT.0) THEN
        KX = 1
    ELSE
        KX = 1 - (LENX-1)*INCX
    END IF
    IF (INCY.GT.0) THEN
        KY = 1
    ELSE
        KY = 1 - (LENY-1)*INCY
    END IF
    IF (BETA.NE.ONE) THEN
        IF (INCY.EQ.1) THEN
            IF (BETA.EQ.ZERO) THEN
                DO 10 I = 1,LENY
                    Y(I) = ZERO
10             CONTINUE
            ELSE
                DO 20 I = 1,LENY
                    Y(I) = BETA*Y(I)
20             CONTINUE
            END IF
        ELSE
            IY = KY
            IF (BETA.EQ.ZERO) THEN
                DO 30 I = 1,LENY
                    Y(IY) = ZERO
                    IY = IY + INCY
30             CONTINUE
            ELSE
                DO 40 I = 1,LENY
                    Y(IY) = BETA*Y(IY)
                    IY = IY + INCY
40             CONTINUE
            END IF
        END IF
    END IF
    IF (ALPHA.EQ.ZERO) RETURN
    IF (TRANS == 'N') THEN
        JX = KX
        IF (INCY.EQ.1) THEN
            DO 60 J = 1,N
                TEMP = ALPHA*X(JX)
                DO 50 I = 1,M
                    Y(I) = Y(I) + TEMP*A(I,J)
50             CONTINUE
                JX = JX + INCX
60         CONTINUE
        ELSE
            DO 80 J = 1,N
                TEMP = ALPHA*X(JX)
                IY = KY
                DO 70 I = 1,M
                    Y(IY) = Y(IY) + TEMP*A(I,J)
                    IY = IY + INCY
70             CONTINUE
                JX = JX + INCX
80         CONTINUE
        END IF
    ELSE
        JY = KY
        IF (INCX.EQ.1) THEN
            DO 100 J = 1,N
                TEMP = ZERO
                DO 90 I = 1,M
                    TEMP = TEMP + A(I,J)*X(I)
90             CONTINUE
                Y(JY) = Y(JY) + ALPHA*TEMP
                JY = JY + INCY
100         CONTINUE
        ELSE
            DO 120 J = 1,N
                TEMP = ZERO
                IX = KX
                DO 110 I = 1,M
                    TEMP = TEMP + A(I,J)*X(IX)
                    IX = IX + INCX
110             CONTINUE
                Y(JY) = Y(JY) + ALPHA*TEMP
                JY = JY + INCY
120         CONTINUE
        END IF
    END IF
    RETURN
END

SUBROUTINE classifier(conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, fc1_w, fc1_b, fc2_w, fc2_b, sequence, classify)
    IMPLICIT NONE
		INTEGER, PARAMETER :: n_channels = 3
		INTEGER, PARAMETER :: window_size = 1500 !128
    INTEGER, PARAMETER :: seq_len = window_size * n_channels
    INTEGER, PARAMETER :: img_len = window_size !seq_len
    INTEGER, PARAMETER :: conv1_out_ch = 64
    INTEGER, PARAMETER :: conv2_out_ch = 64
    INTEGER, PARAMETER :: conv3_out_ch = 64
    INTEGER, PARAMETER :: conv_kernel = 11
    INTEGER, PARAMETER :: stride_length = 5
    INTEGER, PARAMETER :: pool_size = 2
    INTEGER, PARAMETER :: fc1_out = 64
    INTEGER, PARAMETER :: fc2_out = 4
    INTEGER, PARAMETER :: conv1_out_len = img_len - conv_kernel + 1        ! 
    INTEGER, PARAMETER :: conv2_out_len = conv1_out_len - conv_kernel + 1  ! 
    INTEGER, PARAMETER :: conv3_out_len = conv2_out_len - conv_kernel + 1  !
    INTEGER, PARAMETER :: pooled_len = conv3_out_len / pool_size           ! 
    INTEGER, PARAMETER :: fc_in = 64 !GAP !conv3_out_ch * pooled_len

    DOUBLE PRECISION, INTENT(IN) :: conv1_w(conv1_out_ch, n_channels, conv_kernel)  ! (out_channels, in_channels, kernel)
    DOUBLE PRECISION, INTENT(IN) :: conv1_b(conv1_out_ch, 1, 1)
    DOUBLE PRECISION, INTENT(IN) :: conv2_w(conv2_out_ch, conv1_out_ch, conv_kernel)
    DOUBLE PRECISION, INTENT(IN) :: conv2_b(conv2_out_ch, 1, 1)
    DOUBLE PRECISION, INTENT(IN) :: conv3_w(conv3_out_ch, conv2_out_ch, conv_kernel)
    DOUBLE PRECISION, INTENT(IN) :: conv3_b(conv3_out_ch, 1, 1)
    DOUBLE PRECISION, INTENT(IN) :: fc1_w(fc_in, fc1_out, 1)
    DOUBLE PRECISION, INTENT(IN) :: fc1_b(fc1_out, 1, 1)
    DOUBLE PRECISION, INTENT(IN) :: fc2_w(fc1_out, fc2_out, 1)             ! (, )
    DOUBLE PRECISION, INTENT(IN) :: fc2_b(fc2_out, 1, 1)
    DOUBLE PRECISION, INTENT(IN) :: sequence(seq_len)
    DOUBLE PRECISION, INTENT(OUT) :: classify(fc2_out)
		DOUBLE PRECISION :: image(n_channels, window_size)

    DOUBLE PRECISION :: conv1_out(conv1_out_ch, conv1_out_len)
    DOUBLE PRECISION :: conv2_out(conv2_out_ch, conv2_out_len)
		DOUBLE PRECISION :: conv3_out(conv3_out_ch, conv3_out_len)
    DOUBLE PRECISION :: pooled_out(conv3_out_ch, pooled_len)
		DOUBLE PRECISION :: fc1_in(fc_in)
    DOUBLE PRECISION :: conv1_col(n_channels * conv_kernel, conv1_out_len)
    DOUBLE PRECISION :: conv2_col(conv1_out_ch * conv_kernel, conv2_out_len)
    DOUBLE PRECISION :: conv3_col(conv2_out_ch * conv_kernel, conv3_out_len)
		DOUBLE PRECISION :: conv1_w_mat(conv1_out_ch, n_channels * conv_kernel)
		DOUBLE PRECISION :: conv2_w_mat(conv2_out_ch, conv1_out_ch * conv_kernel)
		DOUBLE PRECISION :: conv3_w_mat(conv3_out_ch, conv2_out_ch * conv_kernel)
    DOUBLE PRECISION :: A(fc_in, fc1_out), Y(fc1_out)
    INTEGER :: i, j, k, idx, c

		! --- Reshape flat input sequence into 2D image: (n_channels x window_size)
		! JS.flat() uses row/time-major flattening
		! ie. sequence = [ch0_t0, ch1_t0, ch2_t0, ch0_t1, ch1_t1, ch2_t1, ..., ch0_t127, ch1_t127, ch2_t127]
		! Flip the indexing to reconstruct in FORTRAN column/channel-major formatting
		! ie. image(i, j) = channel i at time j
		!       image(i, j) = sequence((j - 1) * n_channels + i)
		DO j = 1, window_size
			DO i = 1, n_channels
				image(i, j) = sequence((j - 1) * n_channels + i)
			END DO
		END DO

			! ! --- Conv1d layer 1 ---
		! DO i = 1, conv1_out_ch
			! DO j = 1, conv1_out_len
				! conv1_out(i,j) = conv1_b(i,1,1)
				! DO c = 1, n_channels
					! DO k = 1, conv_kernel
						! conv1_out(i,j) = conv1_out(i,j) + conv1_w(i,c,k) * image(c, j + k - 1)
					! END DO
				! END DO
				! IF (conv1_out(i,j) < 0.0d0) conv1_out(i,j) = 0.0d0
			! END DO
		! END DO
		
		! --- Conv1d layer 1 ---
		DO i = 1, conv1_out_ch
			idx = 0
			DO c = 1, n_channels
				DO k = 1, conv_kernel
					idx = idx + 1
					conv1_w_mat(i, idx) = conv1_w(i, c, k)
				END DO
			END DO
		END DO
		! im2col
		DO j = 1, conv1_out_len
			idx = 0
			DO c = 1, n_channels
				DO k = 1, conv_kernel
					idx = idx + 1
					conv1_col(idx, j) = image(c, (j-1)*stride_length + k)
				END DO
			END DO
		END DO
		!DGEMM
		CALL DGEMM('N', 'N', conv1_out_ch, conv1_out_len, n_channels * conv_kernel, 1.0d0, &
					 conv1_w_mat, conv1_out_ch, conv1_col, n_channels * conv_kernel, 0.0d0, &
					 conv1_out, conv1_out_ch)
		! Add bias and ReLU
		DO i = 1, conv1_out_ch
			DO j = 1, conv1_out_len
				conv1_out(i, j) = conv1_out(i, j) + conv1_b(i, 1, 1)
				IF (conv1_out(i, j) < 0.0d0) conv1_out(i, j) = 0.0d0
			END DO
		END DO
		
		! --- Conv1d layer 2 ---
		DO i = 1, conv2_out_ch
			idx = 0
			DO c = 1, conv1_out_ch
				DO k = 1, conv_kernel
					idx = idx + 1
					conv2_w_mat(i, idx) = conv2_w(i, c, k)
				END DO
			END DO
		END DO
		! im2col 
		DO j = 1, conv2_out_len
			idx = 0
			DO c = 1, conv1_out_ch
				DO k = 1, conv_kernel
					idx = idx + 1
					conv2_col(idx, j) = conv1_out(c, (j-1)*stride_length + k)
				END DO
			END DO
		END DO
		! DGEMM
		CALL DGEMM('N', 'N', conv2_out_ch, conv2_out_len, conv1_out_ch * conv_kernel, 1.0d0, &
					 conv2_w_mat, conv2_out_ch, conv2_col, conv1_out_ch * conv_kernel, 0.0d0, &
					 conv2_out, conv2_out_ch)
		! Add bias and ReLU for conv2 output
		DO i = 1, conv2_out_ch
			DO j = 1, conv2_out_len
				conv2_out(i, j) = conv2_out(i, j) + conv2_b(i, 1, 1)
				IF (conv2_out(i, j) < 0.0d0) conv2_out(i, j) = 0.0d0
			END DO
		END DO
		
		! --- Conv1d layer 3 ---
		DO i = 1, conv3_out_ch
			idx = 0
			DO c = 1, conv2_out_ch
				DO k = 1, conv_kernel
					idx = idx + 1
					conv3_w_mat(i, idx) = conv3_w(i, c, k)
				END DO
			END DO
		END DO
		! im2col 
		DO j = 1, conv3_out_len
			idx = 0
			DO c = 1, conv2_out_ch
				DO k = 1, conv_kernel
					idx = idx + 1
					conv3_col(idx, j) = conv2_out(c, (j-1)*stride_length + k)
				END DO
			END DO
		END DO
		! DGEMM
		CALL DGEMM('N', 'N', conv3_out_ch, conv3_out_len, conv2_out_ch * conv_kernel, 1.0d0, &
					 conv3_w_mat, conv3_out_ch, conv3_col, conv2_out_ch * conv_kernel, 0.0d0, &
					 conv3_out, conv3_out_ch)
		! Add bias and ReLU for conv3 output
		DO i = 1, conv3_out_ch
			DO j = 1, conv3_out_len
				conv3_out(i, j) = conv3_out(i, j) + conv3_b(i, 1, 1)
				IF (conv3_out(i, j) < 0.0d0) conv3_out(i, j) = 0.0d0
			END DO
		END DO	
	
    ! --- MaxPool1d with kernel=2, stride=2 (default=kernel) ---
    DO i = 1, conv3_out_ch
        DO j = 1, pooled_len
            pooled_out(i,j) = MAX(conv3_out(i, pool_size*j - 1), conv3_out(i, pool_size*j))
        END DO
    END DO

		! Flatten pooled output into fc1_in vector
			! idx = 0
			! DO i = 1, conv3_out_ch
					! DO j = 1, pooled_len
							! idx = idx + 1
							! fc1_in(idx) = pooled_out(i, j)
					! END DO
			! END DO	
		! GAP
		DO i = 1, conv3_out_ch
				fc1_in(i) = SUM(pooled_out(i, 1:pooled_len)) / REAL(pooled_len, 8) !double
		END DO

    A = fc1_w(:, :, 1)
    Y = fc1_b(1:fc1_out, 1, 1)
    call DGEMV('T', fc_in, fc1_out, 1.0d0, A, fc_in, fc1_in, 1, 1.0d0, Y, 1)
		A(1:fc1_out, 1:fc2_out) = fc2_w(1:fc1_out, 1:fc2_out, 1)
    Y = MAX(0.0d0, Y)
		classify = fc2_b(1:fc2_out, 1, 1)
    call DGEMV('T', fc1_out, fc2_out, 1.0d0, A, fc1_out, Y, 1, 1.0d0, classify, 1)

END SUBROUTINE classifier