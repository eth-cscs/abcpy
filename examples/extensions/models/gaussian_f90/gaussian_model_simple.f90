module gaussian_model
contains
    subroutine gaussian(output, mu, sigma, k, seed)
        integer, intent(in) :: k, seed
        real(8), intent(in) :: mu, sigma
        real(8), intent(out) :: output(k)

        integer :: i, n
        real(8) :: r, theta
        real(8), dimension(:), allocatable :: temp
        integer(4), dimension(:), allocatable :: seed_arr

        ! get random seed array size and fill seed_arr with provided seed
        call random_seed(size = n)
        allocate(seed_arr(n))
        seed_arr = seed
        call random_seed(put = seed_arr)

        ! create 2k random numbers uniformly from [0,1]
        if(allocated(temp)) then
            deallocate(temp)
        end if
        allocate(temp(k * 2))
        call random_number(temp)

        ! Use Box-Muller transfrom to create normally distributed variables
        do i = 1, k
            r = (-2.0 * log(temp(2 * i - 1)))**0.5
            theta = 2 * 3.1415926 * temp(2 * i)
            output(i) = mu + sigma * r * sin(theta)
        end do
    end subroutine gaussian
end module gaussian_model

program main
    use gaussian_model
    implicit none

    integer, parameter :: k = 100
    integer :: seed = 9, i
    real(8) :: mu = 10.0, sigma = 2.0
    real(8) :: output(k)

    call gaussian(output, mu, sigma, k, seed)

    do i = 1, k
        write(*, *) output(i)
    end do
end program main
