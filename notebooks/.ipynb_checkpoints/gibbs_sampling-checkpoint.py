{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "932244b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True, False])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\" Gibbs Sampling.\n",
    "\n",
    "@author: Damian Hoedtke\n",
    "@date: June, 22nd '21\n",
    "\n",
    "\"\"\"\n",
    "from numpy.random import rand\n",
    "from numpy import copy\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# initial distribution\n",
    "state0 = rand(2) > 0.5\n",
    "\n",
    "# MC step\n",
    "def mc_step(state):\n",
    "    \n",
    "    for i, point in enumerate(state):\n",
    "        \n",
    "        # calculate density\n",
    "        mask = np.ones(state.size, dtype=bool)\n",
    "        mask[i] = 0\n",
    "        rho = sum(state[mask]) / state[mask].size\n",
    "    \n",
    "        # sample probability\n",
    "        p = 0.5 * (0.5 + rho)\n",
    "        \n",
    "        # assign new value\n",
    "        state[i] = (rand(1) < p)[0]\n",
    "        \n",
    "    return state\n",
    "    \n",
    "state = copy(state0)\n",
    "mc_step(state)\n",
    "\n",
    "print()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f2189570",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(rand(1) < 0)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0aa4174",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
