{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "23299c16",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([ 29.,  81.,  78., 103., 114., 132., 127., 116.,  90.,  87.,  43.]),\n",
       " array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]),\n",
       " <BarContainer object of 11 artists>)"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\"\"\" Gibbs Sampling.\n",
    "\n",
    "@author: Damian Hoedtke\n",
    "@date: June, 22nd '21\n",
    "\n",
    "\"\"\"\n",
    "from numpy.random import rand, seed\n",
    "from numpy import copy, ones\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "seed(0)\n",
    "\n",
    "# initial distribution\n",
    "n = 10\n",
    "state0 = rand(n) > 0.5\n",
    "\n",
    "# MC step\n",
    "def mc_step(state):\n",
    "    \n",
    "    for i, point in enumerate(state):\n",
    "        \n",
    "        # calculate density\n",
    "        mask = ones(state.size, dtype=bool)\n",
    "        mask[i] = 0\n",
    "        rho = sum(state[mask]) / state[mask].size\n",
    "    \n",
    "        # sample probability\n",
    "        p = 0.3 * 0.5 + 0.7 * rho\n",
    "        \n",
    "        # assign new value\n",
    "        state[i] = (rand(1) < p)[0]\n",
    "        \n",
    "    return state\n",
    "    \n",
    "state = copy(state0)\n",
    "\n",
    "ntrys = 1000\n",
    "nsteps = 500\n",
    "\n",
    "mean_n = []\n",
    "for _ in range(ntrys):\n",
    "    state = rand(n) > 0.5\n",
    "    for _ in range(nsteps):\n",
    "        mc_step(state)\n",
    "    \n",
    "    mean_n.append(sum(state))\n",
    "    \n",
    "plt.hist(mean_n, bins=[i for i in range(n+2)])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "385c7a48",
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
