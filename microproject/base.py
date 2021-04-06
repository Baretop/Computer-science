import numpy
#from pysph.examples._db_geometry import create_2D_filled_region
from pysph.base.utils import get_particle_array
from pysph.sph.iisph import IISPHScheme
from pysph.solver.application import Application

dx = 0.05
hdx = 1.0
rho0 = 1000


class DropFalling(Application):
    def create_particles(self):
        x1 = numpy.arange(1.5, 2.5, dx)
        y1 = numpy.arange(4.2, 5.2, dx)
        z1 = numpy.arange(-0.5, 0.5, dx)
        x1, y1, z1 = numpy.meshgrid(x1, y1, z1)
        x1 = x1.ravel()
        y1 = y1.ravel()
        z1 = z1.ravel()

        #x1, y1 = create_2D_filled_region(1.5, 4.2, 2.5, 5.2, dx)
        #x2, y2 = create_2D_filled_region(0, 0, 4, 4, dx)

        p = ((x1 - 2) ** 2 + (y1 - 4.7) ** 2 + (z1 - 0) ** 2) < 0.4 ** 2
        x1 = x1[p]
        y1 = y1[p]
        z1 = z1[p]

        x2 = numpy.arange(0, 4, dx)
        y2 = numpy.arange(2, 4, dx)
        z2 = numpy.arange(-2, 2, dx)
        x2, y2, z2 = numpy.meshgrid(x2, y2, z2)
        x2 = x2.ravel()
        y2 = y2.ravel()
        z2 = z2.ravel()

        x = numpy.concatenate((x1, x2))
        y = numpy.concatenate((y1, y2))
        z = numpy.concatenate((z1, z2))
        v1 = -3 * numpy.ones_like(y1)
        v2 = numpy.zeros_like(y2)
        v = numpy.concatenate((v1, v2))

        rho = numpy.ones_like(v)*rho0
        h = numpy.ones_like(v)*hdx*dx
        m = numpy.ones_like(v)*dx*dx*dx*rho0

        fluid = get_particle_array(
            name='fluid', x=x, y=y, z=z, v=v, rho=rho, m=m, h=h
        )
        self.scheme.setup_properties([fluid])
        return [fluid]

    def create_scheme(self):
        s = IISPHScheme(fluids=['fluid'], solids=[], dim=3, rho0=rho0)
        return s

    def configure_scheme(self):
        dt = 2e-3
        tf = 2.0
        self.scheme.configure_solver(
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=10
        )


if __name__ == '__main__':
    app = DropFalling()
    app.run()